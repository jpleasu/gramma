#!/usr/bin/env python3
"""
The parser module parses GLF to GExpr objects.

A GExpr is an abstract syntax tree for analysis and interpretation or compilation.

The GrammaGramma object indexes the GExpr of each defined rule and its parameters by name.

"""
import io
import logging
from itertools import groupby
from typing import Dict, List, Set, Union, Optional, Literal, cast, Tuple, IO, Iterator, Callable

import lark

from .util import SetStack

log = logging.getLogger('gramma.parser')

# XXX move this to __init__??
gcode_globals = globals()


def identifier2string(lt: lark.Tree) -> str:
    tok = cast(lark.Token, lt.children[0])
    return tok.value


gramma_grammar = r"""
    ?start : ruledefs

    ruledefs : (ruledef|COMMENT)+

    ruledef : identifier  rule_parms?  ":=" expr ";"
    
    rule_parms : "(" identifier ("," identifier)* ")" 
    
    ?expr : choosein
    
    ?choosein : "choose"? identifier "~" alt ("," identifier "~" alt)* "in" alt | alt

    ?alt : weight? tern ("|" weight? tern)*
    ?weight: number| code
    
    ?tern :  code "?" denoted ":" denoted | denoted
    
    ?denoted : cat ("/" denotation)*
    
    ?denotation : number | string | identifier | code  | dfunc
    dfunc.2 : identifier "(" dfunc_args? ")"
    dfunc_args : denotation ("," denotation)*
    

    ?cat : rep ("." rep)*

    ?rep : atom ( "{" rep_args "}" )?
    rep_args : (INT|code|rep_dist) 
             | (INT|code)? COMMA (INT|code)? (COMMA rep_dist)?
    rep_dist : identifier "(" rep_dist_args ")"
    rep_dist_args : number ("," number)*

    ?atom : string
          | identifier
          | func
          | range
          | "(" expr ")"

    range : "["  range_part ("," range_part )* "]"
    range_part : CHAR  (".." CHAR)?
    
 
    func.2 : identifier "(" func_args? ")"
    func_args : func_arg ("," func_arg)*
    ?func_arg : expr|code|number


    identifier : NAME
    number: INT|FLOAT
    string : STRING|CHAR|LONG_STRING
    code :  CODE
    
    CODE : /`[^`]*`/
    NAME : /[a-zA-Z_][a-zA-Z_0-9]*/
    
    STRING : /[u]?r?("(?!"").*?(?<!\\)(\\\\)*?"|'(?!'').*?(?<!\\)(\\\\)*?')/i
    CHAR.2 : /'([^\\']|\\([\nrt']|x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}))'/
    LONG_STRING.2: /[u]?r?("(?:"").*?(?<!\\)(\\\\)*?"(?:"")|'''.*?(?<!\\)(\\\\)*?''')/is

    COMMENT : /#[^\n]*/
    COMMA: ","

    %import common.WS
    %import common.FLOAT
    %import common.INT

    %ignore COMMENT
    %ignore WS
"""


class GrammaParseError(Exception):
    pass


class LarkTransformer:
    """
        a top-down transformer from lark.Tree to GExpr that handles lexically scoped variables
    """

    vars: Union[Set[str], SetStack[str]]
    rulenames: Set[str]

    def __init__(self, rulenames: Set[str]):
        self.vars = set()
        self.rulenames = rulenames

    def push_vars(self, new_vars):
        self.vars = SetStack(self.vars)
        self.vars.update(new_vars)

    def pop_vars(self):
        self.vars = self.vars.parent

    def visit(self, lt):
        if isinstance(lt, lark.lexer.Token):
            return GTok.from_ltok(lt)

        if hasattr(self, lt.data):
            return getattr(self, lt.data)(lt)
        raise GrammaParseError(f'''unrecognized Lark node "{lt.data}" during parse of glf: {lt}''')

    def string(self, lt):
        return GTok('string', lt.children[0].value)

    def code(self, lt):
        return GCode(lt.children[0].value[1:-1])

    def ruledef(self, lt):
        rname = identifier2string(lt.children[0])
        if len(lt.children) == 3:
            rule_parms = lt.children[1]
            parms = [identifier2string(parm) for parm in rule_parms.children]
        else:
            parms = []
        self.push_vars(parms)
        rhs = self.visit(lt.children[-1])
        self.pop_vars()
        return RuleDef(rname, parms, rhs)

    def choosein(self, lt):
        # lt.children = [var1, expr1, var2, expr2, ..., varN, exprN, child]
        i = iter(lt.children[:-1])
        var_exprs = dict((identifier2string(v), self.visit(d)) for v, d in zip(i, i))

        # push new lexical scope, process child, and pop
        self.push_vars(var_exprs.keys())
        child = self.visit(lt.children[-1])
        self.pop_vars()
        return GChooseIn(var_exprs, child)

    def alt(self, lt):
        weights = []
        children = []
        for clt in lt.children:
            # weights are numbers or code, and children can be neither of these
            if clt.data == 'number':
                weights.append(GTok.from_ltok(clt.children[0]))
            elif clt.data == 'code':
                weights.append(self.code(clt))
            else:
                # weights come before their corresponding child element, so
                # if there's a gap, fill it with a 1
                if len(weights) <= len(children):
                    weights.append(GTok.from_int(1))
                children.append(self.visit(clt))
        return GAlt(weights, children)

    def tern(self, lt):
        code = GCode(lt.children[0].children[0][1:-1])
        return GTern(code, [self.visit(clt) for clt in lt.children[1:]])

    def denoted(self, lt):
        return GDenoted(self.visit(lt.children[0]), self.visit(lt.children[1]))

    def denotation(self, lt):
        print(lt)
        return self.visit(lt.children[0])

    def cat(self, lt):
        return GCat([self.visit(clt) for clt in lt.children])

    def rep(self, lt):
        """
            {,}     - uniform, integer size bounds
            {#}     - exactly #
            {d}     - sample from distribution d
            {,d}    - unclear, disallow
            {#,d}   - silly, disallow
            {,#}    - uniform with upper bound, lower bound is 0
            {#,}    - uniform with lower bound, no upper bound
            {#,#}   - uniform with lower and upper bounds
            {#,,d}  - sample from distribution d, truncate if below lower bound
            {,#,d}  - sample from distribution d, truncate if above upper bound
            {#,#,d} - sample from distribution d, truncate if out of bounds
        """
        child = self.visit(lt.children[0])
        lo: Union[GTok, GCode, None] = None
        hi: Union[GTok, GCode, None] = None
        ncommas = 0
        dist: Optional[RepDist] = None
        for c in lt.children[1].children:
            if isinstance(c, lark.Token) and c.type == 'COMMA':
                ncommas += 1
            elif isinstance(c, lark.Tree) and c.data == 'rep_dist':
                dist = self.visit(c)
            elif ncommas == 0:
                lo = self.visit(c)
            else:
                hi = self.visit(c)

        if ncommas == 0:
            hi = lo

        return GRep(child, lo, hi, dist)

    def range(self, lt):
        # pairs = [  (base, count), (base, count), ... ]
        # where base is ord(starting char) and count is the size of the part
        pairs = []
        for part in lt.children:
            cl = part.children
            base = ord(eval(cl[0].value))
            if len(cl) == 1:
                count = 1
            else:
                e = ord(eval(cl[1].value))
                count = 1 + e - base
            pairs.append((base, count))
        return GRange(pairs)

    def rep_dist(self, lt):
        name = identifier2string(lt.children[0])
        args = [self.visit(clt) for clt in lt.children[1].children]
        return RepDist(name, args)

    def func(self, lt):
        fname = identifier2string(lt.children[0])
        if len(lt.children) > 1:
            fargs = [self.visit(clt) for clt in lt.children[1].children]
        else:
            fargs = []

        if fname in self.rulenames:
            return GRuleRef(fname, fargs)
        else:
            return GFuncRef(fname, fargs)

    def dfunc(self, lt):
        fname = identifier2string(lt.children[0])
        fargs = [self.visit(clt) for clt in lt.children[1].children]
        return GDFuncRef(fname, fargs)

    def number(self, lt):
        tok = lt.children[0]
        return GTok(tok.type, tok.value)

    def identifier(self, lt):
        name = identifier2string(lt)
        if name in self.vars:
            return GVarRef(name)
        elif name in self.rulenames:
            return GRuleRef(name, [])
        else:
            return GFuncRef(name, [])


class GExpr:
    """
        the expression tree for a GLF expression.
    """
    __slots__ = 'parent',

    parent: Optional['GExpr']

    def __init__(self):
        self.parent = None

    def get_code(self) -> List['GCode']:
        """
        get code associated with this element _not_ in the AST
        """
        return []

    def get_ancestor(self, cls):
        p = self.parent
        while p is not None:
            if isinstance(p, cls):
                return p
            p = p.parent
        return p

    def walk(self) -> Iterator['GExpr']:
        yield self

    def is_rule(self, rname=None):
        return False

    def copy(self):
        return None

    def simplify(self):
        """copy self.. caller must ultimately set parent attribute"""
        return self.copy()

    def as_num(self):
        raise GrammaParseError('''only tokens (literal ints and floats) have an as_num method''')

    def as_float(self):
        raise GrammaParseError('''only tokens (literal floats) have an as_float method''')

    def as_int(self):
        raise GrammaParseError('''only tokens (literal ints) have an as_int method''')

    def as_str(self):
        raise GrammaParseError('''only tokens (literal strings) have an as_str method''')


class GTok(GExpr):
    _typemap = {'CHAR': 'string'}
    __slots__ = 'type', 'value', 's'

    type: Literal['INT', 'FLOAT', 'string']
    value: str

    def __init__(self, token_type, value):
        GExpr.__init__(self)
        # self.type='string' if type=='CHAR' else type
        self.type = GTok._typemap.get(token_type, token_type)
        self.value = value
        if self.type == 'string':
            self.s = eval(self.value)

    def copy(self):
        return GTok(self.type, self.value)

    def __str__(self):
        return self.value

    def as_native(self) -> Union[int, float, str]:
        if self.type == 'INT':
            return self.as_int()
        elif self.type == 'FLOAT':
            return self.as_float()
        elif self.type == 'string':
            return self.as_str()
        else:
            raise ValueError('''don't recognize GTok type %s''' % self.type)

    def as_int(self) -> int:
        return int(self.value)

    def as_float(self) -> float:
        return float(self.value)

    def as_str(self) -> str:
        return self.s

    def as_num(self) -> Union[int, float]:
        if self.type == u'INT':
            return int(self.value)
        elif self.type == u'FLOAT':
            return float(self.value)
        else:
            raise GrammaParseError('not a num: %s' % self)

    @staticmethod
    def from_ltok(tok: lark.Token) -> 'GTok':
        return GTok(tok.type, tok.value)

    @staticmethod
    def from_str(s: str) -> 'GTok':
        return GTok('string', repr(s))

    @staticmethod
    def from_int(n: int) -> 'GTok':
        return GTok('INT', str(n))

    @staticmethod
    def from_float(n: float) -> 'GTok':
        return GTok('FLOAT', str(n))

    @staticmethod
    def join(tok_iter) -> 'GTok':
        return GTok.from_str(''.join(t.as_str() for t in tok_iter))

    @staticmethod
    def new_empty() -> 'GTok':
        return GTok.from_str('')


class GInternal(GExpr):
    """
        nodes with GExpr children
    """
    __slots__ = 'children',

    children: List[GExpr]

    def __init__(self, children):
        GExpr.__init__(self)
        self.children = children
        for c in self.children:
            c.parent = self

    def __str__(self):
        return '%s(%s)' % (self.__class__.__name__, ','.join(str(clt) for clt in self.children))

    def walk(self) -> Iterator['GExpr']:
        yield self
        for c in self.children:
            for cc in c.walk():
                yield cc

    def flat_simple_children(self):
        cls = self.__class__
        children = []
        for c in self.children:
            c = c.simplify()
            if isinstance(c, cls):
                children.extend(c.children)
            else:
                children.append(c)
        return children


class GCode(GExpr):
    """
       code expression, e.g. for dynamic alternation weights, ternary expressions, and dynamic repetition
    """
    __slots__ = 'expr',
    expr: str

    def __init__(self, expr):
        super().__init__()
        self.expr = expr

    def __str__(self):
        return '`%s`' % self.expr

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return GCode('(%s)+%f' % (self.expr, other))
        return GCode('(%s)+(%s)' % (self.expr, other.expr))

    def copy(self):
        return GCode(self.expr)


class GChooseIn(GInternal):
    """
        choose <var> ~ <dist> in  <child>

        when entering a choosein, the sampler samples dist and saves the result
        in the sampler under "var" ..

        then child is sampled, var is treated like a string-valued rule.

    """
    __slots__ = 'vnames',

    vnames: List[str]

    def __init__(self, var_dists: Dict[str, GExpr], child: GExpr):
        GInternal.__init__(self, list(var_dists.values()) + [child])
        self.vnames = list(var_dists.keys())

    @property
    def dists(self):
        return self.children[:-1]

    @property
    def child(self):
        return self.children[-1]

    @property
    def values(self):
        return self.children[:-1]

    def copy(self):
        return GChooseIn(dict(zip(self.vnames, self.dists)), self.child)

    def __str__(self):
        var_dists = ', '.join('%s~%s' % (var, dist) for var, dist in zip(self.vnames, self.dists))
        return 'choose %s in %s' % (var_dists, self.child)


class GTern(GInternal):
    __slots__ = 'code',

    def __init__(self, code, children):
        GInternal.__init__(self, children)
        self.code = code

    def get_code(self) -> List['GCode']:
        return [self.code]

    def compute_case(self, x):
        return self.code.invoke(x)

    def __str__(self):
        return '%s ? %s : %s' % (self.code, self.children[0], self.children[1])

    def simplify(self):
        return GTern(self.code, [c.simplify() for c in self.children])

    def copy(self):
        return GTern(self.code, [c.copy() for c in self.children])


class GAlt(GInternal):
    __slots__ = 'weights', 'dynamic', 'nweights'

    weights: List[Union[GTok, GCode]]  # numbers or code
    dynamic: bool

    def __init__(self, weights, children):
        GInternal.__init__(self, children)
        self.weights = weights
        self.dynamic = any(isinstance(w, GCode) for w in self.weights)

    def get_code(self) -> List['GCode']:
        return [w for w in self.weights if isinstance(w, GCode)]

    def __str__(self):
        weights = []
        for w in self.weights:
            if isinstance(w, GTok) and w.type == 'INT' and w.as_int() == 1:
                weights.append('')
            else:
                weights.append(str(w) + ' ')
        s = '|'.join('%s%s' % (w, c) for w, c in zip(weights, self.children))

        if self.parent is not None:  # and isinstance(self.parent, (GCat, GRep)):
            return '(%s)' % s
        return s

    def simplify(self):
        if self.dynamic:
            return self.simplify_dynamic()
        return self.simplify_nondynamic()

    def simplify_dynamic(self):
        """
        complicated normalizing factors could make the result less simple..

            `f1` (`g1` a | `g2` b) | `f2` (`g3` c | `g4` d)

            `f1*g1/(g1+g2)` a | `f1*g2/(g1+g2)` b | `f2*g3/(g3+g4)` c | `f2*g4/(g3+g4)` d

        """
        return self.copy()

    def simplify_nondynamic(self):
        weights = []
        children = []

        for w, c in zip(self.weights, self.children):
            c = c.simplify()
            if isinstance(c, GAlt) and not c.dynamic:
                t = sum(c.weights)
                weights.extend([float(w.as_num() * cw.as_num()) / t for cw in c.weights])
                children.extend(c.children)
            else:
                weights.append(w.as_num())
                children.append(c)
        if len(children) == 0:
            return GTok.new_empty()
        if len(children) == 1:
            return children[0]

        # dedupe (and sort) by string representation
        nchildren = []
        nweights = []
        for sc, tups in groupby(sorted(((str(c), c, w) for w, c in zip(weights, children)), key=lambda tup: tup[0]),
                                key=lambda tup: tup[0]):
            tups = list(tups)
            nweights.append(sum(tup[2] for tup in tups))
            nchildren.append(tups[0][1])

        if len(nchildren) == 0:
            return GTok.new_empty()
        if len(nchildren) == 1:
            return nchildren[0]
        return GAlt([GTok.from_float(w) for w in nweights], nchildren)

    def copy(self):
        return GAlt(self.weights, [c.copy() for c in self.children])


class GDenoted(GInternal):
    """
    a denoted expression
    """

    def __init__(self, left, right):
        GInternal.__init__(self, [left, right])

    @property
    def left(self):
        return self.children[0]

    @property
    def right(self):
        return self.children[1]

    def __str__(self):
        s = str(self.left) + '/' + str(self.right)
        if self.parent is not None and isinstance(self.parent, GRep):
            return '(%s)' % s
        return s

    def copy(self):
        return GDenoted(self.left, self.right)

    def simplify(self):
        return self.copy()


class GCat(GInternal):
    def __str__(self):
        s = '.'.join(str(cge) for cge in self.children)
        if self.parent is not None and isinstance(self.parent, GRep):
            return '(%s)' % s
        return s

    def copy(self):
        return GCat([c.copy() for c in self.children])

    def simplify(self):
        children = self.flat_simple_children()
        if len(children) == 0:
            return GTok.new_empty()
        if len(children) == 1:
            return children[0]

        nchildren = []
        for t, cl in groupby(children, lambda c: isinstance(c, GTok)):
            if t:
                nchildren.append(GTok.join(cl))
            else:
                nchildren.extend(cl)
        if len(nchildren) == 1:
            return nchildren[0]
        return GCat(nchildren)


class RepDist:
    __slots__ = 'name', 'args'
    name: str
    args: List[GTok]  # list of number tokens

    def __init__(self, name, args):
        self.name = name
        self.args = args

    def __str__(self):
        return f"{self.name}({','.join(str(x) for x in self.args)})"


class GRep(GInternal):
    __slots__ = 'lo', 'hi', 'dist'
    lo: Union[GTok, GCode]
    hi: Union[GTok, GCode]
    dist: Optional[RepDist]

    def __init__(self, child, lo, hi, dist):
        GInternal.__init__(self, [child])
        self.lo = lo
        self.hi = hi
        self.dist = dist

    def get_code(self) -> List['GCode']:
        return [c for c in [self.lo, self.hi] if isinstance(c, GCode)]

    @property
    def child(self):
        return self.children[0]

    def copy(self):
        return GRep([self.child.copy()], self.lo, self.hi, self.dist)

    def simplify(self):
        return GRep([self.child.simplify()], self.lo, self.hi, self.dist)

    def args_str(self):
        if self.lo == self.hi:
            if self.lo is None:
                return ','
            return '%s' % self.lo
        lo = '' if self.lo is None else '%s' % self.lo
        hi = '' if self.hi is None else '%s' % self.hi
        return '%s,%s' % (lo, hi)

    def __str__(self):
        child = self.child
        # display x{,,dist} as x{dist}
        if self.lo is None and self.hi is None and self.dist is not None:
            return '%s{%s}' % (child, self.dist)
        # no dist
        if self.dist is None:
            return '%s{%s}' % (child, self.args_str())
        return '%s{%s,%s}' % (child, self.args_str(), self.dist)


class GRange(GExpr):
    __slots__ = 'pairs',
    pairs: List[Tuple[int, int]]

    def __init__(self, pairs):
        """
        pairs - [ (base, count), ...]
            where base is ord(char) and count is the size of the part
        """
        GExpr.__init__(self)
        self.pairs = pairs

    @property
    def chars(self):
        chars = []
        for base, count in self.pairs:
            chars.extend(chr(base + i) for i in range(count))
        return chars

    def copy(self):
        return GRange(self.pairs)

    def simplify(self):
        if len(self.pairs) == 1 and self.pairs[0][1] == 1:
            o = self.pairs[0][0]
            return GTok.from_str(chr(o))
        return self.copy()

    def __str__(self):
        parts = []
        for base, count in sorted(self.pairs):
            if count == 1:
                parts.append("'%s'" % chr(base))
            else:
                parts.append("'%s'..'%s'" % (chr(base), chr(base + count - 1)))

        return '[%s]' % (','.join(parts))


class GFuncRef(GInternal):
    """
    a reference to a gfunc. The implementation is in the sampler.
    """
    __slots__ = 'fname',
    fname: str

    def __init__(self, fname, fargs):
        GInternal.__init__(self, fargs)
        self.fname = fname

    def copy(self):
        return self.__class__(self.fname, [c.copy() for c in self.fargs])

    def simplify(self):
        return self.__class__(self.fname, [c.simplify() for c in self.fargs])

    @property
    def fargs(self):
        return self.children

    def __str__(self):
        return '%s(%s)' % (self.fname, ','.join(str(a) for a in self.fargs))


class GDFuncRef(GFuncRef):
    """
    a reference to a dfunc.  These differ from GFuncs only by what their arguments are allowed to be and where
    they show up (the RHS of GDenoted nodes)
    """
    pass


class GRuleRef(GInternal):
    """
    a reference to a rule. The rule definition is part of the GrammaGrammar class.
    """

    __slots__ = 'rname', 'rhs'

    def __init__(self, rname, rargs, rhs=None):
        GInternal.__init__(self, rargs)
        self.rname = rname
        self.rhs = rhs

    def copy(self):
        return GRuleRef(self.rname, self.rargs, self.rhs)

    def is_rule(self, rname=None):
        if rname is None:
            return True
        return self.rname == rname

    @property
    def rargs(self):
        return self.children

    def __str__(self):
        if len(self.rargs) > 0:
            return '%s(%s)' % (self.rname, ','.join(str(c) for c in self.rargs))
        else:
            return self.rname


class GVarRef(GExpr):
    """
    a variable reference. Refers to either a choosein binding, or rule parameter.
    """

    __slots__ = 'vname',

    def __init__(self, vname):
        GExpr.__init__(self)
        self.vname = vname

    def copy(self):
        return GVarRef(self.vname)

    def __str__(self):
        return self.vname


class RuleDef:
    __slots__ = 'rname', 'params', 'rhs'

    def __init__(self, rname: str, params: List[str], rhs: GExpr):
        self.rname = rname
        self.params = params
        self.rhs = rhs


class GrammaGrammar:
    GLF_PARSER = lark.Lark(gramma_grammar, parser='earley', lexer='standard', start='start')
    GEXPR_PARSER = lark.Lark(gramma_grammar, parser='earley', lexer='standard', start='expr')

    ruledefs: Dict[str, RuleDef]

    def __init__(self, glf: str):
        self.ruledefs = {}

        lark_tree: lark.Tree = GrammaGrammar.GLF_PARSER.parse(glf)
        ruledef_trees = cast(List[lark.Tree], lark_tree.children)
        rulenames = set([identifier2string(cast(lark.Tree, ruledef.children[0])) for ruledef in ruledef_trees])
        transformer = LarkTransformer(rulenames)

        for ruledef_ast in lark_tree.children:
            ruledef = transformer.visit(ruledef_ast)
            self.ruledefs[ruledef.rname] = ruledef

    def walk(self) -> Iterator[GExpr]:
        for ruledef in self.ruledefs.values():
            for ge in ruledef.rhs.walk():
                yield ge

    @staticmethod
    def of(grammar: Union[str, IO[str], 'GrammaGrammar']) -> 'GrammaGrammar':
        if isinstance(grammar, GrammaGrammar):
            return grammar
        elif isinstance(grammar, str):
            return GrammaGrammar(grammar)
        elif isinstance(grammar, io.TextIOBase):
            return GrammaGrammar(grammar.read())
        raise TypeError('unknown type for grammar')

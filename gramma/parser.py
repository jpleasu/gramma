#!/usr/bin/env python3
"""
The parser module parses GLF to GExpr objects.

A GExpr is an abstract syntax tree for analysis and interpretation or compilation.

The GrammaGramma object indexes the GExpr of each defined rule and its parameters by name.

"""
import io
import logging
from itertools import groupby
from typing import Dict, List, Set, Union, Optional, Literal, cast, Tuple, IO, Iterator, ClassVar, Callable, TypeVar, \
    Type

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
    
    ?denotation : number | string | identifier | code | dfunc
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


# noinspection PyMethodMayBeStatic
class LarkTransformer:
    """
        a top-down transformer from lark.Tree to GExpr that handles lexically scoped variables
    """

    vars: SetStack[str]
    rulenames: Set[str]

    def __init__(self, rulenames: Set[str]):
        self.vars = SetStack()
        self.rulenames = rulenames

    def visit(self, lt: lark.Tree) -> 'GExpr':
        if hasattr(self, lt.data):
            f = cast(Callable[[lark.Tree], GExpr], getattr(self, lt.data))
            return f(lt)
        raise GrammaParseError(f'''unrecognized Lark node "{lt.data}" during parse of glf: {lt}''')  # pragma: no cover

    def string(self, lt):
        return GTok('string', lt.children[0].value)

    def code(self, lt: lark.Tree) -> 'GCode':
        t = cast(lark.Token, lt.children[0])
        return GCode(t.value[1:-1]).loc(Location.from_lark_token(t))

    def ruledef(self, lt: lark.Tree) -> 'RuleDef':
        rname = identifier2string(cast(lark.Tree, lt.children[0]))
        parms: List[str]
        if len(lt.children) == 3:
            rule_parms = cast(lark.Tree, lt.children[1])
            parms = [identifier2string(cast(lark.Tree, parm)) for parm in rule_parms.children]
        else:
            parms = []

        with self.vars.context(parms):
            rhs = self.visit(cast(lark.Tree, lt.children[-1]))
        return RuleDef(rname, parms, rhs).loc(Location.from_lark_tree(lt))

    def choosein(self, lt):
        # lt.children = [var1, expr1, var2, expr2, ..., varN, exprN, child]
        i = iter(lt.children[:-1])
        var_exprs: Dict[str, GExpr] = dict((identifier2string(v), self.visit(d)) for v, d in zip(i, i))

        with self.vars.context(var_exprs.keys()):
            child = self.visit(lt.children[-1])
        return GChooseIn(var_exprs, child).loc(Location.from_lark_tree(lt))

    def alt(self, lt):
        weights: List[Union[GTok, GCode]] = []
        children: List[GExpr] = []
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
        return GAlt(weights, children).loc(Location.from_lark_tree(lt))

    def tern(self, lt):
        code = GCode(lt.children[0].children[0][1:-1])
        return GTern(code, [self.visit(clt) for clt in lt.children[1:]]).loc(Location.from_lark_tree(lt))

    def denoted(self, lt: lark.Tree) -> 'GDenoted':
        children = cast(List[lark.Tree], lt.children)
        return GDenoted(self.visit(children[0]), self.visit(children[1])).loc(Location.from_lark_tree(lt))

    def cat(self, lt: lark.Tree) -> 'GCat':
        return GCat([self.visit(cast(lark.Tree, clt)) for clt in lt.children]).loc(Location.from_lark_tree(lt))

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
                dist = self.rep_dist(c)
            else:
                n: Union[GTok, GCode]
                if isinstance(c, lark.Token):
                    n = GTok.from_ltok(c)
                else:
                    n = self.code(c)

                if ncommas == 0:
                    lo = n
                else:
                    hi = n

        if ncommas == 0:
            hi = lo

        return GRep(child, lo, hi, dist).loc(Location.from_lark_tree(lt))

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
        return GRange(pairs).loc(Location.from_lark_tree(lt))

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
            return GRuleRef(fname, fargs).loc(Location.from_lark_tree(lt))
        else:
            return GFuncRef(fname, fargs).loc(Location.from_lark_tree(lt))

    def dfunc(self, lt):
        fname = identifier2string(lt.children[0])
        fargs: List[GExpr]
        if len(lt.children) > 1:
            fargs = [self.visit(clt) for clt in lt.children[1].children]
        else:
            fargs = []
        return GDFuncRef(fname, fargs).loc(Location.from_lark_tree(lt))

    def number(self, lt):
        tok = lt.children[0]
        return GTok(tok.type, tok.value).loc(Location.from_lark_tree(lt))

    def identifier(self, lt):
        name = identifier2string(lt)
        if name in self.vars:
            return GVarRef(name).loc(Location.from_lark_tree(lt))
        elif name in self.rulenames:
            return GRuleRef(name, []).loc(Location.from_lark_tree(lt))
        else:
            raise GrammaParseError(f'no variable named "{name}" in scope')


class Location:
    line: int
    column: int

    def __init__(self, line: int, column: int, pos: int):
        self.line = line
        self.column = column
        self.pos = pos

    @staticmethod
    def from_lark_token(tok: lark.Token) -> 'Location':
        return Location(tok.line, tok.column, tok.pos_in_stream)

    @staticmethod
    def from_lark_tree(lt: lark.Tree) -> 'Location':
        return Location(lt.meta.line, lt.meta.column, lt.meta.start_pos)


GExprT = TypeVar('GExprT', bound='GExpr')


class GExpr:
    """
        the expression tree for a GLF expression.
    """
    __slots__ = 'parent', 'location'

    parent: Optional['GExpr']
    location: Optional[Location]

    def __init__(self):
        self.parent = None
        self.location = None

    def loc(self: GExprT, location: Optional[Location]) -> GExprT:
        self.location = location
        return self

    def locstr(self) -> str:
        if self.location is None:
            raise GrammaParseError('node missing location information')  # pragma: no cover
        return f'line {self.location.line}, column {self.location.column}'

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

    def is_ruleref(self, rname=None):
        return False

    def copy(self: GExprT) -> GExprT:  # pragma: no cover
        ...

    def simplify(self: GExprT) -> GExprT:
        return self.copy()

    def with_parens(self, s: str, if_equals: bool = False) -> str:
        if self.parent is not None:
            p = GEXPR_PRECEDENCE_MAP[self.parent.__class__]
            c = GEXPR_PRECEDENCE_MAP[self.__class__]
            if if_equals and c <= p or c < p:
                return '(' + s + ')'
        return s


class GTok(GExpr):
    _typemap: Dict[str, Literal['string']] = {'CHAR': 'string'}
    __slots__ = 'type', 'value', 's'

    TypeEnum = Literal['INT', 'FLOAT', 'string']
    type: TypeEnum
    value: str
    s: str

    def __init__(self, token_type: str, value: str):
        GExpr.__init__(self)
        self.type = GTok._typemap.get(token_type, cast(GTok.TypeEnum, token_type))
        self.value = value
        if self.type == 'string':
            self.s = eval(self.value)

    def copy(self):
        return GTok(self.type, self.value).loc(self.location)

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
            raise GrammaParseError(f'''unknown GTok type {self.type}''')  # pragma: no cover

    def as_int(self) -> int:
        return int(self.value)

    def as_float(self) -> float:
        return float(self.value)

    def as_str(self) -> str:
        if not hasattr(self, 's'):
            return '(None)'
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
        return GTok(tok.type, tok.value).loc(Location.from_lark_token(tok))

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
    def join(tok_iter: Iterator['GTok']) -> 'GTok':
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

    def __str__(self):  # pragma: no cover
        return '%s(%s)' % (self.__class__.__name__, ','.join(str(clt) for clt in self.children))

    def walk(self, upto: Optional[Callable[[GExpr], bool]] = None) -> Iterator['GExpr']:
        yield self
        if upto is not None:
            if upto(self):
                return
        for c in self.children:
            yield from c.walk()

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

    def __init__(self, expr: str):
        GExpr.__init__(self)
        self.expr = expr

    def __str__(self):
        return '`%s`' % self.expr

    def copy(self):
        return GCode(self.expr).loc(self.location)


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
        return GChooseIn(dict(zip(self.vnames, self.dists)), self.child).loc(self.location)

    def __str__(self):
        var_dists = ', '.join('%s~%s' % (var, dist) for var, dist in zip(self.vnames, self.dists))
        return 'choose %s in %s' % (var_dists, self.child)


class GTern(GInternal):
    __slots__ = 'code',

    code: GCode

    def __init__(self, code: GCode, children: List[GExpr]):
        GInternal.__init__(self, children)
        self.code = code

    def get_code(self) -> List['GCode']:
        return [self.code]

    def __str__(self):
        return '%s?%s:%s' % (self.code, self.children[0], self.children[1])

    def simplify(self):
        return GTern(self.code.copy(), [c.simplify() for c in self.children])

    def copy(self):
        return GTern(self.code.copy(), [c.copy() for c in self.children]).loc(self.location)


def defloat(x: Union[int, float]) -> Union[int, float]:
    """ 1.0 -> 1,  2.3 -> 2.3"""
    if isinstance(x, int):
        return x
    i = int(x)
    if i == x:
        return i
    return x


class GAlt(GInternal):
    __slots__ = 'weights', 'dynamic', 'nweights'

    weights: List[Union[GTok, GCode]]  # numbers or code
    dynamic: bool

    def __init__(self, weights: List[Union[GTok, GCode]], children: List[GExpr]):
        GInternal.__init__(self, children)
        self.weights = weights
        self.dynamic = any(isinstance(w, GCode) for w in self.weights)

    def get_code(self) -> List['GCode']:
        return [w for w in self.weights if isinstance(w, GCode)]

    def __str__(self):
        weights = []
        for w in self.weights:
            if isinstance(w, GTok) and (w.type == 'INT' and w.as_int() == 1 or
                                        w.type == 'FLOAT' and w.as_float() == 1.0):
                weights.append('')
            else:
                weights.append(str(w) + ' ')
        s = '|'.join('%s%s' % (w, c) for w, c in zip(weights, self.children))

        return self.with_parens(s, if_equals=True)

    def simplify(self):
        if self.dynamic:
            return self.simplify_dynamic()
        return self.simplify_nondynamic()

    def simplify_dynamic(self):
        """
        complicated normalizing factors could make the obvious thing much less simple..

            `f1` (`g1` a | `g2` b) | `f2` (`g3` c | `g4` d)

            `f1*g1/(g1+g2)` a | `f1*g2/(g1+g2)` b | `f2*g3/(g3+g4)` c | `f2*g4/(g3+g4)` d

        """
        return self.copy()

    def simplify_nondynamic(self):
        weights = []
        children = []

        # flatten
        self_weights = cast(List[GTok], self.weights)
        mult: Union[int, float] = 1
        for w, c in zip(self_weights, self.children):
            if w.as_num() == 0:
                continue
            c = c.simplify()
            if isinstance(c, GAlt) and not c.dynamic:
                c_weights = cast(List[GTok], c.weights)
                t = sum(cw.as_num() for cw in c_weights)
                mult *= t
                weights.extend([float(w.as_num() * cw.as_num()) / t for cw in c_weights])
                children.extend(c.children)
            else:
                weights.append(w.as_num())
                children.append(c)
        weights = [w * mult for w in weights]
        if len(children) == 0:
            return GTok.new_empty()
        if len(children) == 1:
            return children[0]

        # dedupe (and sort) by string representation
        nchildren = []
        nweights = []
        for sc, tupsit in groupby(sorted(((str(c), c, w) for w, c in zip(weights, children)), key=lambda tup: tup[0]),
                                  key=lambda tup: tup[0]):
            tups = list(tupsit)
            nweights.append(sum(tup[2] for tup in tups))
            nchildren.append(tups[0][1])

        nweights = [defloat(w) for w in nweights]

        if len(nchildren) == 1:
            return nchildren[0]
        return GAlt([GTok.from_float(w) for w in nweights], nchildren)

    def copy(self):
        return GAlt(self.weights, [c.copy() for c in self.children]).loc(self.location)


class GDenoted(GInternal):
    """
    a denoted expression
    """

    def __init__(self, left, right):
        GInternal.__init__(self, [left, right])

    @property
    def left(self) -> GExpr:
        return self.children[0]

    @property
    def right(self) -> GExpr:
        return self.children[1]

    def __str__(self):
        s = str(self.left) + '/' + str(self.right)
        return self.with_parens(s)

    def copy(self):
        return GDenoted(self.left.copy(), self.right.copy()).loc(self.location)

    def simplify(self):
        return GDenoted(self.left.simplify(), self.right.simplify()).loc(self.location)


class GCat(GInternal):
    def __str__(self):
        if len(self.children) == 0:
            s = "''"
        else:
            s = '.'.join(str(cge) for cge in self.children)
        return self.with_parens(s)

    def copy(self):
        return GCat([c.copy() for c in self.children]).loc(self.location)

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
    default: ClassVar['RepDist']
    name: str
    args: List[GTok]  # list of number tokens

    def __init__(self, name, args):
        self.name = name
        self.args = args

    def __str__(self):
        return f"{self.name}({','.join(str(x) for x in self.args)})"


RepDist.default = RepDist('uniform', [])


class GRep(GInternal):
    __slots__ = 'lo', 'hi', 'dist'
    lo: Union[GTok, GCode, None]
    hi: Union[GTok, GCode, None]
    dist: RepDist

    def __init__(self, child: GExpr, lo: Union[GTok, GCode, None], hi: Union[GTok, GCode, None],
                 dist: Optional[RepDist]):
        GInternal.__init__(self, [child])
        self.lo = lo
        self.hi = hi
        self.dist = RepDist.default if dist is None else dist

    def get_code(self) -> List['GCode']:
        return [c for c in [self.lo, self.hi] if isinstance(c, GCode)]

    @property
    def child(self):
        return self.children[0]

    def copy(self):
        return GRep(self.child.copy(), self.lo, self.hi, self.dist).loc(self.location)

    def simplify(self):
        return GRep(self.child.simplify(), self.lo, self.hi, self.dist)

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
        if self.lo is None and self.hi is None and self.dist is not RepDist.default:
            return '%s{%s}' % (child, self.dist)
        # default dist
        if self.dist is RepDist.default:
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
        chars: List[str] = []
        for base, count in self.pairs:
            chars.extend(chr(base + i) for i in range(count))
        return chars

    def copy(self):
        return GRange(self.pairs).loc(self.location)

    def simplify(self):
        if len(self.pairs) == 1 and self.pairs[0][1] == 1:
            o = self.pairs[0][0]
            return GTok.from_str(chr(o))
        pairs = []
        base, count = self.pairs[0]
        for b, c in self.pairs[1:]:
            if b == base + count:
                count += c
            else:
                pairs.append((base, count))
                base, count = b, c
        pairs.append((base, count))
        return GRange(pairs).loc(self.location)

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
        return self.__class__(self.fname, [c.copy() for c in self.fargs]).loc(self.location)

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

    __slots__ = 'rname',
    rname: str

    def __init__(self, rname, rargs):
        GInternal.__init__(self, rargs)
        self.rname = rname

    def copy(self):
        return GRuleRef(self.rname, self.rargs).loc(self.location)

    def is_ruleref(self, rname=None):
        if rname is None:
            return True
        return self.rname == rname

    @property
    def rargs(self) -> List[GExpr]:
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
        return GVarRef(self.vname).loc(self.location)

    def __str__(self):
        return self.vname


GEXPR_PRECEDENCE_MAP: Dict[Type['GExpr'], int] = {
    GTok: 6,
    GVarRef: 6,
    GDFuncRef: -1,  # avoid parenthesizing arguments of functions
    GFuncRef: -1,
    GRuleRef: -1,

    GRep: 5,
    GCat: 4,
    GDenoted: 3,
    GTern: 2,
    GAlt: 1,
    GChooseIn: 0,
}


class RuleDef:
    __slots__ = 'rname', 'params', 'rhs', 'location'

    rname: str
    rhs: GExpr
    params: List[str]

    location: Optional[Location]

    def __init__(self, rname: str, params: List[str], rhs: GExpr):
        self.rname = rname
        self.params = params
        self.rhs = rhs
        self.location = None

    def loc(self, location: Location) -> 'RuleDef':
        self.location = location
        return self

    def locstr(self) -> str:
        if self.location is None:
            raise GrammaParseError('ruledef missing location information')  # pragma: no cover
        return f'line {self.location.line}, column {self.location.column}'


class GrammaGrammar:
    GLF_PARSER = lark.Lark(gramma_grammar, parser='earley', lexer='standard', start='start', propagate_positions=True)
    GEXPR_PARSER = lark.Lark(gramma_grammar, parser='earley', lexer='standard', start='expr', propagate_positions=True)

    ruledefs: Dict[str, RuleDef]

    def __init__(self, glf_text: str):
        lark_tree: lark.Tree = GrammaGrammar.GLF_PARSER.parse(glf_text)
        ruledef_trees = cast(List[lark.Tree], lark_tree.children)
        rulenames = set([identifier2string(cast(lark.Tree, ruledef.children[0])) for ruledef in ruledef_trees])

        transformer = LarkTransformer(rulenames)
        self.ruledefs = {}
        for ruledef_ast in cast(List[lark.Tree], lark_tree.children):
            ruledef = cast(RuleDef, transformer.visit(ruledef_ast))
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

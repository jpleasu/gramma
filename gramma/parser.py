#!/usr/bin/env python3
"""
"""
import copy
import logging
import numbers
import sys
from functools import reduce
from itertools import groupby
from typing import Dict, Tuple, List, Set

import lark
import numpy as np

from .util import SetStack

try:
    import astpretty
except ImportError:
    pass

log = logging.getLogger('gramma.parser')

# XXX move this to __init__??
gcode_globals = globals()


def identifier2string(lt: lark.Tree) -> str:
    return lt.children[0].value


gramma_grammar = r"""
    ?start : ruledefs

    ruledefs : (ruledef|COMMENT)+

    ruledef : identifier  rule_parms?  ":=" expr ";"
    
    rule_parms : "(" identifier ("," identifier)* ")" 
    
    ?expr : choosein
    
    ?choosein : "choose"? identifier "~" alt ("," identifier "~" alt)* "in" alt | alt

    ?alt : weight? tern ("|" weight? tern)*
    ?weight: number| code
    
    ?tern :  code "?" den ":" den | den
    
    ?den : cat ("/" cat)*

    ?cat : rep ("." rep)*

    ?rep: atom ( "{" rep_args "}" )?
    rep_args : (INT|code)? (COMMA (INT|code)?)? (COMMA func)?
            | func

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


class LarkTransformer(object):
    """
        a top-down transformer from lark.Tree to GExpr that handles lexically scoped variables
    """

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
        if lt.data == 'string':
            return GTok('string', lt.children[0].value)
        if lt.data == 'code':
            return GCode(lt.children[0].value[1:-1])

        if hasattr(self, lt.data):
            return getattr(self, lt.data)(lt)
        raise GrammaParseError('''unrecognized Lark node %s during parse of glf''' % lt)

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
            if clt.data == 'number':
                weights.append(GTok.from_ltok(clt.children[0]).as_num())
                continue
            if clt.data == 'code':
                weights.append(GCode(clt.children[0][1:-1]))
                continue
            if len(weights) <= len(children):
                weights.append(1)
            children.append(self.visit(clt))
        return GAlt(weights, children)

    def tern(self, lt):
        code = GCode(lt.children[0].children[0][1:-1])
        return GTern(code, [self.visit(clt) for clt in lt.children[1:]])

    def den(self, lt):
        return GDen(self.visit(lt.children[0]), self.visit(lt.children[1]))

    def cat(self, lt):
        return GCat([self.visit(clt) for clt in lt.children])

    def rep(self, lt):
        """
            {,}     - uniform, integer size bounds
            {#}     - exactly #
            {d}     - sample from distribution d
            {,d}    - " " " "
            {,#}    - uniform with upper bound, lower bound is 0
            {#,}    - uniform with lower bound, no upper bound
            {#,#}   - uniform with lower and upper bounds
            {#,d}   - sample from distribution d, reject if below lower bound
            {,#,d}  - sample from distribution d, reject if above upper bound
            {#,#,d} - sample from distribution d, reject if out of bounds
        """
        child = self.visit(lt.children[0])
        args = [self.visit(c) for c in lt.children[1].children[:]]
        # figure out the distribution.. if not a GCode or a GFunc, assume uniform
        a = args[-1]
        if isinstance(a, GFunc):
            fname = a.fname
            fargs = [x.as_num() for x in a.fargs]
            dist = RepDist(a.fname, fargs)
            if fname == u'geom':
                # "a"{geom(n)} has an average of n copies of "a"
                parm = 1 / float(fargs[0] + 1)
                g = lambda x: x.random.geometric(parm) - 1
            elif fname == 'norm':
                g = lambda x: int(x.random.normal(*fargs) + .5)
            elif fname == 'binom':
                g = lambda x: x.random.binomial(*fargs)
            elif fname == 'choose':
                g = lambda x: x.random.choice(fargs)
            else:
                raise GrammaParseError('no dist %s' % (fname))

            del args[-2:]
        else:
            dist = RepDist('unif', [])
            g = None

        # parse bounds to lo and hi, each is either None, GTok integer, or GCode
        if len(args) == 0:
            # {`dynamic`}
            lo = None
            hi = None
        elif len(args) == 1:
            # {2} or {2,`dynamic`}.. where the latter is pretty stupid
            lo = hi = args[0]
        elif len(args) == 2:
            # {0,,`dynamic`}
            if (str(args[1]) == ','):
                lo = args[0].as_int()
                hi = None
            else:
                # {,2} or {,2,`dynamic`}
                if str(args[0]) != ',':
                    raise GrammaParseError('expected comma in repetition arg "%s"' % lt)
                lo = None
                hi = args[1]
        elif len(args) == 3:
            # {2,3} or {2,3,`dynamic`}
            lo = args[0]
            hi = args[2]
        else:
            raise GrammaParseError('failed to parse repetition arg "%s"' % lt)

        if hi is None:
            if lo is None:
                if g is None:
                    rgen = lambda x: x.random.randint(0, 2 ** 32)
                else:
                    rgen = g
            else:
                if isinstance(lo, GCode):
                    if g is None:
                        rgen = lambda x: x.random.randint(lo.invoke(x), 2 ** 32)
                    else:
                        rgen = lambda x: max(lo.invoke(x), g(x))
                else:
                    lo = lo.as_int()
                    if g is None:
                        rgen = lambda x: x.random.randint(lo, 2 ** 32)
                    else:
                        rgen = lambda x: max(lo, g(x))
        else:
            if isinstance(hi, GCode):
                if lo is None:
                    if g is None:
                        rgen = lambda x: x.random.randint(0, 1 + hi.invoke(x))
                    else:
                        rgen = lambda x: min(g(x), hi.invoke(x))
                else:
                    if isinstance(lo, GCode):
                        if g is None:
                            rgen = lambda x: x.random.randint(lo.invoke(x), 1 + hi.invoke(x))
                        else:
                            rgen = lambda x: max(lo.invoke(x), min(g(x), hi.invoke(x)))
                    else:
                        lo = lo.as_int()
                        if g is None:
                            rgen = lambda x: x.random.randint(lo, 1 + hi.invoke(x))
                        else:
                            rgen = lambda x: max(lo, min(g(x), hi.invoke(x)))
            else:
                hi = hi.as_int()
                hip1 = 1 + hi
                if lo is None:
                    if g is None:
                        rgen = lambda x: x.random.randint(0, hip1)
                    else:
                        rgen = lambda x: min(g(x), hi)
                else:
                    if isinstance(lo, GCode):
                        if g is None:
                            rgen = lambda x: x.random.randint(lo.invoke(x), hip1)
                        else:
                            rgen = lambda x: max(lo.invoke(x), min(g(x), hi))
                    else:
                        lo = lo.as_int()
                        if g is None:
                            rgen = lambda x: x.random.randint(lo, hip1)
                        else:
                            rgen = lambda x: max(lo, min(g(x), hi))

        # lo and hi, are each either None, int, or GCode
        return GRep([child], lo, hi, rgen, dist)

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

    def func(self, lt):
        fname = identifier2string(lt.children[0])
        if len(lt.children) > 1:
            fargs = [self.visit(clt) for clt in lt.children[1].children]
        else:
            fargs = []

        if fname in self.rulenames:
            return GRule(fname, fargs)
        else:
            return GFunc(fname, fargs)

    def number(self, lt):
        tok = lt.children[0]
        return GTok(tok.type, tok.value)

    def identifier(self, lt):
        name = identifier2string(lt)
        if name in self.vars:
            return GVar(name)
        elif name in self.rulenames:
            return GRule(name, [])
        else:
            return GFunc(name, [])

class GExpr(object):
    """
        the expression tree for a GLF expression.
    """
    __slots__ = 'parent',

    def __init__(self):
        self.parent = None

    def get_code(self):
        return []

    def get_ancestor(self, cls):
        p = self.parent
        while p is not None:
            if isinstance(p, cls):
                return p
            p = p.parent
        return p

    def is_rule(self, rname=None):
        return False

    def copy(self):
        return None

    def simplify(self):
        """copy self.. caller must ultimately set parent attribute"""
        return self.copy()

    def as_num(self):
        raise GrammaParseError('''only tokens (literal numbers) have an as_num method''')

    def as_float(self):
        raise GrammaParseError('''only tokens (literal ints) have an as_float method''')

    def as_int(self):
        raise GrammaParseError('''only tokens (literal ints) have an as_int method''')

    def as_str(self):
        raise GrammaParseError('''only tokens (literal strings) have an as_str method''')


class GTok(GExpr):
    _typemap = {'CHAR': 'string'}
    __slots__ = 'type', 'value', 's'

    def __init__(self, type, value):
        GExpr.__init__(self)
        # self.type='string' if type=='CHAR' else type
        self.type = GTok._typemap.get(type, type)
        self.value = value
        if self.type == 'string':
            self.s = eval(self.value)

    def copy(self):
        return GTok(self.type, self.value)

    def __str__(self):
        return self.value

    def as_native(self):
        if self.type == 'INT':
            return self.as_int()
        elif self.type == 'FLOAT':
            return self.as_float()
        elif self.type == 'string':
            return self.as_str()
        else:
            raise ValueError('''don't recognize GTok type %s''' % self.type)

    def as_int(self):
        return int(self.value)

    def as_float(self):
        return float(self.value)

    def as_str(self):
        return self.s

    def as_num(self):
        if self.type == u'INT':
            return int(self.value)
        elif self.type == u'FLOAT':
            return float(self.value)
        else:
            raise GrammaParseError('not a num: %s' % self)

    @staticmethod
    def from_ltok(lt):
        return GTok(lt.type, lt.value)

    @staticmethod
    def from_str(s):
        return GTok('string', repr(s))

    @staticmethod
    def join(tok_iter):
        return GTok.from_str(''.join(t.as_str() for t in tok_iter))

    @staticmethod
    def new_empty():
        return GTok.from_str('')


class GInternal(GExpr):
    """
        nodes with GExpr children
    """
    __slots__ = 'children',

    def __init__(self, children):
        GExpr.__init__(self)
        self.children = children
        for c in self.children:
            c.parent = self

    def __str__(self):
        return '%s(%s)' % (self.__class__.__name__, ','.join(str(clt) for clt in self.children))

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
    __slots__ = 'expr', 'compiled'

    def __init__(self, expr):
        self.expr = expr
        self.compiled = None  # set in finalize_gexpr

    def invoke(self, x):
        locs = dict(x.params.__dict__, **x.state.__dict__)
        return eval(self.compiled, gcode_globals, locs)

    def __str__(self):
        return '`%s`' % self.expr

    def __add__(self, other):
        if isinstance(other, numbers.Number):
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

    def __init__(self, var_dists, child):
        GInternal.__init__(self, list(var_dists.values()) + [child])
        self.vnames = list(var_dists.keys())

    @property
    def dists(self):
        return self.children[:-1]

    @property
    def child(self):
        return self.children[-1]

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

    def get_code(self):
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

    def __init__(self, weights, children):
        GInternal.__init__(self, children)
        self.dynamic = any(w for w in weights if isinstance(w, GCode))

        if self.dynamic:
            self.weights = weights
        else:
            self.weights = weights
            w = np.array(weights)
            self.nweights = w / w.sum()

    def get_code(self):
        return [w for w in self.weights if isinstance(w, GCode)]

    def compute_weights(self, x):
        """
            dynamic weights are computed using the SamplerInterface variable
            every time an alternation is invoked.
        """
        if self.dynamic:
            w = np.array([w.invoke(x) if isinstance(w, GCode) else w for w in self.weights])
            return w / w.sum()
        return self.nweights

    def __str__(self):
        weights = []
        for w in self.weights:
            if isinstance(w, GCode):
                weights.append('`%s` ' % w.expr)
            elif w == 1:
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
                weights.extend([float(w * cw) / t for cw in c.weights])
                children.extend(c.children)
            else:
                weights.append(w)
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
        return GAlt(nweights, nchildren)

    def copy(self):
        return GAlt(self.weights, [c.copy() for c in self.children])


class GDen(GInternal):
    """
    denotations
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
        return GDen(self.left, self.right)

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

        l = []
        for t, cl in groupby(children, lambda c: isinstance(c, GTok)):
            if t:
                l.append(GTok.join(cl))
            else:
                l.extend(cl)
        if len(children) == 1:
            return children[0]
        return GCat(l)


class RepDist(object):
    __slots__ = 'name', 'args'

    def __init__(self, name, args):
        self.name = name
        self.args = args

    def __str__(self):
        return f"{self.name}({','.join(str(x) for x in self.args)})"


class GRep(GInternal):
    __slots__ = 'rgen', 'lo', 'hi', 'dist'

    def __init__(self, children, lo, hi, rgen, dist):
        GInternal.__init__(self, children)
        self.lo = lo
        self.hi = hi
        self.rgen = rgen
        self.dist = dist

    def get_code(self):
        return [c for c in [self.lo, self.hi] if isinstance(c, GCode)]

    @property
    def child(self):
        return self.children[0]

    def copy(self):
        return GRep([self.child.copy()], self.lo, self.hi, self.rgen, self.dist)

    def simplify(self):
        return GRep([self.child.simplify()], self.lo, self.hi, self.rgen, self.dist)

    def range_args(self):
        if self.lo == self.hi:
            if self.lo is None:
                return ','
            return '%s' % (self.lo)
        lo = '' if self.lo is None else '%s' % (self.lo)
        hi = '' if self.hi is None else '%s' % (self.hi)
        return '%s,%s' % (lo, self.hi)

    def __str__(self):
        child = self.child
        if self.dist.name == 'unif':
            return '%s{%s}' % (child, self.range_args())
        return '%s{%s,%s}' % (child, self.range_args(), self.dist)


class GRange(GExpr):
    __slots__ = 'pairs',

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


class GFunc(GInternal):
    """
    a reference to a gfunc. The gfunc implementation should be in the sampler.
    """
    __slots__ = 'fname', 'gf'

    def __init__(self, fname, fargs, gf=None):
        GInternal.__init__(self, fargs)
        self.fname = fname
        # set in finalize_gexpr
        self.gf = gf

    def copy(self):
        return GFunc(self.fname, [c.copy() for c in self.fargs], self.gf)

    def simplify(self):
        return GFunc(self.fname, [c.simplify() for c in self.fargs], self.gf)

    @property
    def fargs(self):
        return self.children

    def __str__(self):
        return '%s(%s)' % (self.fname, ','.join(str(a) for a in self.fargs))


class GRule(GInternal):
    """
    a reference to a rule. The rule definition is part of the GrammaGrammar class.
    """

    __slots__ = 'rname', 'rhs'

    def __init__(self, rname, rargs, rhs=None):
        GInternal.__init__(self, rargs)
        self.rname = rname
        self.rhs = rhs

    def copy(self):
        return GRule(self.rname, self.rargs, self.rhs)

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


class GVar(GExpr):
    """
    a variable reference. Refers to either a choosein binding, or rule parameter.
    """

    __slots__ = 'vname',

    def __init__(self, vname):
        GExpr.__init__(self)
        self.vname = vname

    def copy(self):
        return GRule(self.vname)

    def __str__(self):
        return self.vname

class GrammaGrammar(object):
    GLF_PARSER = lark.Lark(gramma_grammar, parser='earley', lexer='standard', start='start')
    GEXPR_PARSER = lark.Lark(gramma_grammar, parser='earley', lexer='standard', start='expr')

    ruledefs: Dict[str, Tuple[GExpr, List[str]]]

    def __init__(self, glf: str):
        self.ruledefs = {}

        lark_tree = GrammaGrammar.GLF_PARSER.parse(glf)
        rulenames = set([identifier2string(ruledef.children[0]) for ruledef in lark_tree.children])
        transformer = LarkTransformer(rulenames)

        for ruledef in lark_tree.children:
            rulename = identifier2string(ruledef.children[0])
            if len(ruledef.children) == 3:
                rule_parms = ruledef.children[1]
                parms = [identifier2string(parm) for parm in rule_parms.children]
            else:
                parms = []
            transformer.push_vars(parms)
            self.ruledefs[rulename] = (transformer.visit(ruledef.children[-1]), parms)
            transformer.pop_vars()

#!/usr/bin/env python3
r"""
    TODO:
        - resampling
            - only ever resample with a bias-to-previous
            - generate an "unrolled" grammar
            - "replay" is encoded in unrolled grammar
            - finish design for unrolling api for gexprs
        - simplify gfuncs
            - move gfuncs to sampler
            - non-generator gfuncs, functions of strings, same as C++
        - simplify state
            - no state object, just sampler object
            - add save_state, load_state methods of sampler
            - move analysis out of GrammaGrammar, make it optional
        - Monte Carlo with StackWatcher to estimate excessive loops / other
          over-represented constructs.

"""
import ast
import builtins
import copy
import inspect
import logging
import numbers
import sys
import textwrap
from collections import deque
from functools import reduce
from itertools import groupby

import lark
import numpy as np
from six import string_types

from gramma import pysa
from .util import SetStack, DictStack

try:
    import astpretty
except ImportError:
    pass

# sys.setrecursionlimit(20000)
sys.setrecursionlimit(200000)

logging.basicConfig(
    format='%(asctime)-15s.%(msecs)d [%(name)s]: %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO)
log = logging.getLogger('gramma')

gramma_grammar = r"""
    ?start : ruledefs

    ruledefs : (ruledef|COMMENT)+

    ruledef : NAME ":=" expr ";"

    ?expr : choosein
    
    ?choosein : "choose"? NAME "~" tern ("," NAME "~" tern)* "in" tern | tern

    ?tern :  code "?" alt ":" alt | alt

    ?alt : weight? den ("|" weight? den)*
    
    ?den : cat ("->" cat)?

    ?cat : rep ("." rep)*

    ?rep: atom ( "{" rep_args "}" )?

    rep_args : (INT|code)? (COMMA (INT|code)?)? (COMMA func)?
            | func

    ?atom : string
         | identifier
         | func
         | range
         | "(" expr ")"

    identifier : NAME

    func.2 : NAME "(" func_args? ")"

    func_args : func_arg ("," func_arg)*

    ?func_arg : code|alt|INT|FLOAT

    ?weight: number| code

    number: INT|FLOAT

    
    range_part : CHAR  (".." CHAR)?
    range : "["  range_part ("," range_part )* "]"
    
    NAME : /[a-zA-Z_][a-zA-Z_0-9]*/

    string : STRING|CHAR|LONG_STRING

    code : /`[^`]*`/

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

    def __init__(self):
        self.vars = {}

    def push_vars(self, new_vars):
        self.vars = SetStack()
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
        # lt.children = [var1, dist1, var2, dist2, ..., varN, distN, child]
        i = iter(lt.children[:-1])
        var_dists = dict((v.value, self.visit(d)) for v, d in zip(i, i))

        # push new lexical scope, process child, and pop
        self.push_vars(var_dists.keys())
        child = self.visit(lt.children[-1])
        self.pop_vars()
        return GChooseIn(var_dists, child)

    def tern(self, lt):
        code = GCode(lt.children[0].children[0][1:-1])
        return GTern(code, [self.visit(clt) for clt in lt.children[1:]])

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
        fname = lt.children[0].value

        if len(lt.children) > 1:
            fargs = [self.visit(clt) for clt in lt.children[1].children]
        else:
            fargs = []

        return GFunc(fname, fargs)

    def identifier(self, lt):
        name = lt.children[0].value
        if name in self.vars:
            return GVar(name)
        else:
            return GRule(name)


class GExpr(object):
    """
        the expression tree for a GLF expression.
    """
    __slots__ = 'parent',

    def __init__(self):
        self.parent = None

    def get_code(self):
        return []

    def get_meta(self):
        return GExprMetadata.DEFAULT

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

    def dump_meta(ge, out=sys.stdout, indent=0):
        """recursive meta dumper"""
        print((' ' * indent) + '%s[%s]' % (ge, ge.get_meta()), file=out)

        if isinstance(ge, GInternal):
            for c in ge.children:
                c.dump_meta(out, indent + 1)

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


gcode_globals = globals()


class GCode(GExpr):
    """
       code expression, e.g. for dynamic alternation weights, ternary expressions, and dynamic repetition
    """
    __slots__ = 'expr', 'compiled', 'meta'

    def __init__(self, expr, meta=None):
        self.expr = expr
        self.compiled = None  # set in finalize_gexpr
        self.meta = GExprMetadata.DEFAULT.copy() if meta is None else meta

    def invoke(self, x):
        locs = dict(x.params.__dict__, **x.state.__dict__)
        return eval(self.compiled, gcode_globals, locs)

    def __str__(self):
        return '`%s`' % self.expr

    def __add__(self, other):
        if isinstance(other, numbers.Number):
            return GCode('(%s)+%f' % (self.expr, other))
        return GCode('(%s)+(%s)' % (self.expr, other.expr))

    def get_meta(self):
        return self.meta

    def copy(self):
        return GCode(self.expr, self.meta.copy())


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

    def get_meta(self):
        return self.code.meta

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

    def get_meta(self):
        return reduce(lambda a, b: a | b, (w.meta for w in self.get_code()), GExprMetadata(uses_random=True))

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
        s = str(self.left) + '->' + str(self.right)
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

    def get_meta(self):
        return reduce(lambda a, b: a | b, (c.meta for c in self.get_code()), GExprMetadata(uses_random=True))

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

    def get_meta(self):
        return GExprMetadata.DEFAULT_RANDOM

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
    __slots__ = 'fname', 'gf'

    def __init__(self, fname, fargs, gf=None):
        GInternal.__init__(self, fargs)
        self.fname = fname
        # set in finalize_gexpr
        self.gf = gf

    def get_meta(self):
        return self.gf.meta

    def copy(self):
        return GFunc(self.fname, [c.copy() for c in self.fargs], self.gf)

    def simplify(self):
        return GFunc(self.fname, [c.simplify() for c in self.fargs], self.gf)

    @property
    def fargs(self):
        return self.children

    def __str__(self):
        return '%s(%s)' % (self.fname, ','.join(str(a) for a in self.children or self.fargs))


class GRule(GExpr):
    """
    a _reference_ to a rule.. the rule definition is part of the GrammaGrammar class.
    """

    __slots__ = 'rname', 'rhs'

    def __init__(self, rname, rhs=None):
        GExpr.__init__(self)
        self.rname = rname
        self.rhs = rhs

    def copy(self):
        return GRule(self.rname, self.rhs)

    def is_rule(self, rname=None):
        if rname is None:
            return True
        return self.rname == rname

    def __str__(self):
        return self.rname


class GVar(GExpr):
    """
    a _reference_ to a variable.
    """

    __slots__ = 'vname',

    def __init__(self, vname):
        GExpr.__init__(self)
        self.vname = vname

    def copy(self):
        return GRule(self.vname)

    def __str__(self):
        return self.vname


class GExprMetadata(object):
    """
    state and rand metadata
    """
    __slots__ = 'statevar_defs', 'statevar_uses', 'uses_random', 'samples'

    def __init__(self, statevar_defs=None, statevar_uses=None, samples=False, uses_random=False):
        self.statevar_defs = set() if statevar_defs is None else statevar_defs
        self.statevar_uses = set() if statevar_uses is None else statevar_uses
        self.samples = samples
        self.uses_random = uses_random

    def __str__(self):
        pl = []
        if len(self.statevar_defs) > 0:
            pl.append('defs=%s' % ','.join(sorted(self.statevar_defs)))
        if len(self.statevar_uses) > 0:
            pl.append('uses=%s' % ','.join(sorted(self.statevar_uses)))
        if self.uses_random:
            pl.append('uses_random')
        return ' '.join(pl)

    def copy(self):
        return GExprMetadata(copy.deepcopy(self.statevar_defs), copy.deepcopy(self.statevar_uses), self.samples,
                             self.uses_random)

    def __or__(self, other):
        return GExprMetadata(self.statevar_defs | other.statevar_defs,
                             self.statevar_uses | other.statevar_uses,
                             self.samples | other.samples, self.uses_random | other.uses_random)


GExprMetadata.DEFAULT = GExprMetadata()
GExprMetadata.DEFAULT_RANDOM = GExprMetadata(uses_random=True)


class GFuncWrap(object):
    __slots__ = 'f', 'fname', 'analyzer', 'meta'

    def __init__(self, f, fname=None, analyzer=None, meta=None):
        self.f = f
        self.fname = fname
        self.analyzer = analyzer
        self.meta = GExprMetadata.DEFAULT.copy() if meta is None else meta

    def __call__(self, *l, **kw):
        return self.f(*l, **kw)

    def __str__(self):
        return 'gfunc %s%s %s' % (self.fname, ' %s' % self.analyzer if self.analyzer is not None else '', self.meta)

    def copy(self):
        return GFuncWrap(self.f, self.fname, self.analyzer, self.meta.copy())


def gfunc(*args, **kw):
    """
        GrammaGrammar function decorator.

        To extend GLF, annotate methods of a GrammaGrammar child class with
        @gfunc.

        A gfunc
            *) has prototype
                    f(x,args*)
                where
                    x is the gfunc interface:
                        x.state - a GrammaState object for managing all state
                        x.random - a GrammaRandom object for all entropy

                    args are the arguments of f as GExpr elements.  In
                    particular, "constants" are are type GTok and must be
                    converted, and generate GEXpr objects can be sampled from.

            *) mustn't access global variables
            *) may store state as fields of the GrammaState instance, state
            *) mustn't take additional keyword arguments, only "grammar",
                and "state" are allowed.
            *) may sample from GExpr arguments by yielding a GExpr.. the result
                will be a string.
            *) may access entropy from the SamplerInterface instance

        The SamplerInterface state fields are referred to as "state vars".  By
        tracking their use in gfuncs, Gramma can optimize certain operations.

        gfunc decorator keyword arguments

            fname = string
                provides a name other than the method name to use in glf

            analyzer = static function taking a single GFunc argument
                if set, disables autoanalysis, it's called whenever gfunc is
                parsed. The analyzer should populate the GFunc's metadata.

            statevar_defs = list/set
            statevar_uses = list/set
                manual override for automatically inferred statevar def/use

            uses_random = True/False
                manual override for use of random
    """

    def _decorate(f, **kw):
        fname = kw.pop('fname', f.__name__)
        analyzer = kw.pop('analyzer', None)
        return GFuncWrap(f, fname, analyzer, GExprMetadata(**kw))

    if len(args) == 0 or not callable(args[0]):
        return lambda f: _decorate(f, *args, **kw)

    f = args[0]
    return _decorate(f, **kw)


def analyzer_use_first_arg(ge):
    ge.gf.meta.statevar_uses.add(ge.fargs[0].as_str())


def analyzer_def_first_arg(ge):
    ge.gf.meta.statevar_defs.add(ge.fargs[0].as_str())


def get_reset_variable_names(cls, method_name='reset_state'):
    """extract the names of variables defined in reset_state."""
    m = pysa.get_method(cls, method_name)
    if m is None:
        return set()
    state_id = pysa.ast_argname(m.args.args[1])
    return set([n[1:] for n in pysa.get_defs(m) if n[0] == state_id])


class GrammaGrammar(object):
    """
        The class defining functions and state management for a gramma and
        extensions.

        e.g.
         g=GrammaGrammar('start:="a"|"b";')
         sampler=GrammaSampler(g)

         while True:
             print(sampler.sample())
    """

    ruledef_parser = lark.Lark(gramma_grammar, parser='earley', lexer='standard', start='start')
    expr_parser = lark.Lark(gramma_grammar, parser='earley', lexer='standard', start='tern')

    __slots__ = 'sideeffects', 'ruledefs', 'funcdefs', 'reset_states', 'allowed_global_ids', 'param_ids', '_compilemap'

    def __init__(self, gramma_expr_str, sideeffects=None, param_ids=None, allowed_global_ids=None):
        """
            gramma_expr_str - defines the grammar, including a start rule.

            sideeffects - a list of SideEffect objects or classes which the
            grammar, GFunc implementations or GCode expressions, require.

            param_ids - a list of param names that will be ignored by the GFunc
            and GCode analyzers.

            allowed_global_ids - a list of global variable identifiers ignored
            by GFunc and GCode analyzers.
        """
        self._init_compilemap()

        if sideeffects is None:
            sideeffects = []

        # instantiate sideeffect classes
        self.sideeffects = [sideeffect() if inspect.isclass(sideeffect) else sideeffect for sideeffect in sideeffects]

        # analyze sideeffect state variable access
        reset_states = set()
        for se in self.sideeffects:
            reset_states |= se.get_reset_states()

        # analyze reset_state
        reset_states |= self.get_reset_states()

        cls = self.__class__
        allowed_global_ids = [] if allowed_global_ids is None else allowed_global_ids
        param_ids = [] if param_ids is None else param_ids

        self.funcdefs = {}
        for n, gf in inspect.getmembers(self, predicate=lambda x: isinstance(x, GFuncWrap)):
            # make a grammar-local copy of gf
            gf = gf.copy()

            if gf.analyzer is None:
                GFuncAnalyzer(cls, gf, reset_states, param_ids=param_ids, allowed_global_ids=allowed_global_ids)

            self.funcdefs[gf.fname] = gf

        # record metadata
        self.reset_states = reset_states
        self.allowed_global_ids = allowed_global_ids
        self.param_ids = param_ids

        self.ruledefs = {}
        self.add_rules(gramma_expr_str)

    def add_rules(self, ruledef_str):
        lt = GrammaGrammar.ruledef_parser.parse(ruledef_str)
        for ruledef in lt.children:
            lp = LarkTransformer()
            self.ruledefs[ruledef.children[0].value] = lp.visit(ruledef.children[1])
        for ge in self.ruledefs.values():
            self.finalize_gexpr(ge)

    def get_reset_states(self):
        return get_reset_variable_names(self.__class__)

    def reset_state(self, state):
        pass

    @gfunc
    def save_rand(x, slot):
        x.random.save(slot.as_str())
        yield ''

    @gfunc(uses_random=True)
    def load_rand(x, slot):
        x.random.load(slot.as_str())
        yield ''

    @gfunc(uses_random=True)
    def reseed_rand(x):
        x.random.seed(None)
        yield ''

    @gfunc(analyzer=analyzer_use_first_arg)
    def save(x, n, slot):
        x.state.save(n.as_str(), slot.as_str())
        yield ''

    @gfunc(analyzer=analyzer_def_first_arg)
    def load(x, n, slot):
        x.state.load(n.as_str(), slot.as_str())
        yield ''

    @gfunc(fname='def', analyzer=analyzer_def_first_arg)
    def def_(x, n, v):
        if isinstance(v, GTok):
            v = v.as_native()
        elif isinstance(v, GCode):
            v = v.invoke(x)
        setattr(x.state, n.as_str(), v)
        yield ''

    def finalize_gexpr(self, ge):
        """
            grammar dependent finalization of a gexpr:
                1) compute meta for GCode nodes
                2) lookup GFuncs
                3) lookup rules

        """
        if isinstance(ge, GFunc):
            if ge.gf is not None:
                # already finalized
                return
            gf = self.funcdefs.get(ge.fname, None)
            if gf is None:
                raise GrammaAnalysisException('no gfunc named %s available to %s' % (ge.fname, self.__class__.__name__))
            ge.gf = gf
            if gf.analyzer is not None:
                ge.gf = ge.gf.copy()
                gf.analyzer(ge)

        elif isinstance(ge, GRule):
            if ge.rhs is not None:
                # already finalized
                return
            rhs = self.ruledefs.get(ge.rname, None)
            if rhs is None:
                raise GrammaAnalysisException('no rule named %s available to %s' % (ge.rname, self.__class__.__name__))
            ge.rhs = rhs

        elif isinstance(ge, GCode):
            if ge.compiled is None:
                ge.compiled = compile(ge.expr, '<GCode>', 'eval')
            GCodeAnalyzer(self, ge)

        # dynamic elements keep their code outside of the expr tree
        for code in ge.get_code():
            self.finalize_gexpr(code)

        if isinstance(ge, GInternal):
            for c in ge.children:
                self.finalize_gexpr(c)
        return ge

    def parse(self, gramma_expr_str):
        """ gramma expression -> GExpr"""
        lp = LarkTransformer()
        ge = lp.visit(GrammaGrammar.expr_parser.parse(gramma_expr_str))
        self.finalize_gexpr(ge)
        return ge

    def _init_compilemap(self):
        self._compilemap = {}
        for t in [GTok, GChooseIn, GTern, GAlt, GDen, GCat, GRep, GRange, GRule, GVar, GFunc]:
            self._compilemap[t] = getattr(self, 'compile_' + t.__name__)

    def compile_GTok(self, ge):
        def g(x):
            yield ge.s

        return g

    def compile_GChooseIn(self, ge):
        vnames = ge.vnames
        dists = ge.dists
        child = ge.child

        def g(x):
            # sample each variable
            values = []
            for dist in dists:
                values.append((yield dist))
            x.push_vars(dict(zip(vnames, values)))
            # execute child while vname is set
            s = (yield child)
            x.pop_vars()
            yield s

        return g

    def compile_GTern(self, ge):
        def g(x):
            s = yield (ge.children[0] if ge.compute_case(x) else ge.children[1])
            yield s

        return g

    def compile_GAlt(self, ge):
        if ge.dynamic:
            def g(x):
                s = yield x.random.pchoice(ge.children, p=ge.compute_weights(x))
                yield s
        else:
            def g(x):
                s = yield x.random.pchoice(ge.children, p=ge.nweights)
                yield s
        return g

    def compile_GDen(self, ge):
        def g(x):
            s = yield ge.left
            _ = yield ge.right
            yield s

        return g

    def compile_GCat(self, ge):
        def g(x):
            s = ''
            for cge in ge.children:
                s += yield cge
            yield s

        return g

    def compile_GRep(self, ge):
        def g(x):
            s = ''
            n = ge.rgen(x)
            while n > 0:
                s += yield ge.child
                n -= 1
            yield s

        return g

    def compile_GRange(self, ge: GRange):
        chars = ge.chars

        def g(x):
            yield x.random.choice(chars)

        return g

    def compile_GRange_noexpand(self, ge: GRange):
        """
        XXX somewhere between this and compile_GRange is a decent compromise between size and speed
        """
        pairs = ge.pairs
        ind = np.arange(len(pairs))
        w = np.array([count for base, count in pairs], dtype=float)
        w /= w.sum()
        fs = []
        for base, count in pairs:
            if count == 1:
                fs.append(lambda x: chr(base))
            else:
                fs.append(lambda x: chr(base + x.random.randint(0, count)))

        if len(pairs) == 1:
            f = fs[0]

            def g(x):
                yield f(x)
        else:
            def g(x):
                i = x.random.pchoice(ind, w)
                yield fs[i](x)

        return g

    def compile_GRule(self, ge):
        rhs = ge.rhs

        def g(x):
            s = yield (rhs)
            yield s

        return g

    def compile_GVar(self, ge):
        vname = ge.vname

        def g(x):
            yield x.vars[vname]

        return g

    def compile_GFunc(self, ge):
        def g(x, gf=ge.gf, fargs=ge.fargs):
            return gf(x, *fargs)

        return g

    def compile(self, ge):
        return self._compilemap[ge.__class__](ge)


class SideEffect(object):
    """
    Base class for sampler sideeffects.
    """

    __slots__ = ()

    def get_reset_states(self):
        return get_reset_variable_names(self.__class__)

    def reset_state(self, state):
        """initialize managed variables before execution."""
        pass

    def push(self, x, ge):
        """
        Handle state machine push.

        when an expression is compiled to a coroutine, the stack machine
        pushes.  the return value is pushed at the same position in the stack,
        made available to pop later.
        """
        return None

    def pop(self, x, w, s):
        """
        Handle state machine pop.

        when a coroutine completes, returning a string, s, it is popped.  w is
        the value returned by the corresponding push.
        """
        pass


class Transformer(object):
    """
    transforms gexprs prior to coroutine compilation in by the sampler.
    """

    __slots__ = ()

    def transform(self, x, ge):
        """
        x is the SamplerInterface, ge is the incoming GExpr

        must return a gexpr.

        note: cache results!
        """
        return ge


class GrammaAnalysisException(Exception):
    pass


class GFuncAnalyzer(pysa.VariableAccesses):
    allowed_globals = ['struct', 'True', 'False', 'None'] + [x for x in dir(builtins) if x.islower()]

    def __init__(self, target_class, gf, reset_states, param_ids=None, allowed_global_ids=None):
        pysa.VariableAccesses.__init__(self)
        self.target_class = target_class
        self.reset_states = reset_states

        self.allowed_ids = set(GFuncAnalyzer.allowed_globals)
        if allowed_global_ids is not None:
            self.allowed_ids.update(allowed_global_ids)
        self.param_ids = [] if param_ids is None else param_ids
        self.uses_random = False
        self.samples = False
        self.has_terminal_yield = False

        self.statevar_defs = set()
        self.statevar_uses = set()

        s = inspect.getsource(gf.f)
        s = textwrap.dedent(s)
        gf_ast = ast.parse(s).body[0]

        self.fname = gf_ast.name

        if len(gf_ast.args.defaults) > 0:
            self._raise('''gfuncs mustn't use default argument values''')
        if gf_ast.args.kwarg is not None:
            self._raise('''gfuncs mustn't take keyword arguments''')

        fargs = [pysa.NamePath(a) for a in gf_ast.args.args]
        self.iface_id = fargs[0]
        self.extra_args = set(a.s for a in fargs[1:])

        self.allowed_ids.update(a.s for a in fargs)

        self.run(gf_ast)

        # .. and annotate GFuncWrap objects with state and rand metadata
        gf.meta.statevar_defs.update(self.statevar_defs)
        gf.meta.statevar_uses.update(self.statevar_uses)
        gf.meta.uses_random = self.uses_random
        gf.meta.samples = self.samples

        if not self.has_terminal_yield:
            self._raise('''doesn't yield a value''')

        # astpretty.pprint(gf_ast)
        # print(gf.meta)

    def is_iface_id(self, n):
        """n is a NamePath"""
        return n[0].s == self.iface_id

    def _raise(self, msg):
        if hasattr(self, 'stack') and len(self.stack) > 0:
            raise GrammaAnalysisException('''in line %d of gfunc %s of class %s: %s''' % (
                self.stack[-1].lineno, self.fname, self.target_class.__name__, msg))
        raise GrammaAnalysisException('''gfunc %s of class %s: %s''' % (self.fname, self.target_class.__name__, msg))

    def defs(self, n, v):
        if self.is_iface_id(n):
            if n[1].s == 'state':
                self.statevar_defs.add(n[2].s)
            elif n[1].s == 'random' or n[1].s == 'params':
                self._raise('forbidden access to SamplerInterface %s' % n[1:])
            else:
                self._raise(
                    'unexpected SamplerInterface field "%s", only "random", "state", and "params" are accessible' % n[
                                                                                                                    1:].s)
        else:
            self.allowed_ids.add(n.s)

    def uses(self, n):
        if self.is_iface_id(n):
            if n[1].s == 'state':
                nn = n[2:]
                for s in self.reset_states:
                    if nn.begins(s):
                        self.statevar_uses.add(n[2].s)
                        break
                else:
                    self._raise(
                        '%s used without being initialized in any reset_state or explicitly allowed in allowed_global_ids or param_ids' % n.s)
            elif n[1].s == 'params':
                if n[2] == '[]':
                    self._raise(
                        'params is not indexed, define ids with the param_ids argument of GrammaGrammar and set with update_params method of GrammaSampler')
                if not n[2] in self.param_ids:
                    self._raise('param "%s" not declared by grammar' % n[2].s)
            elif n[1].s == 'random':
                self._raise('direct use of random object?')
            else:
                self._raise(
                    'unexpected SamplerInterface field "%s", only "random", "state", and "params" are accessible' % n[
                                                                                                                    1:].s)
        else:
            for p in n.prefixes:
                if p.s in self.allowed_ids:
                    return

            # astpretty.pprint(self.stack[-2])
            self._raise('forbidden access to variable "%s"' % n.s)

    def mods(self, n, v):
        self.uses(n)
        self.defs(n, v)

    def calls(self, n, v):
        if self.is_iface_id(n):
            if n[1].s == 'state':
                self.mods(n, v)
            elif n[1].s == 'random':
                self.uses_random = True
            else:
                self._raise('forbidden all to variable "%s"' % n.s)

    def lambdas(self, l):
        pass

    def visit_Yield(self, y):
        self.visit(y.value)

        if any(n for n in self.stack[:-1] if isinstance(n, (ast.GeneratorExp, ast.ListComp))):
            self._raise('yield in a generator expression or list comprehension')

        p = self.stack[-2]
        if isinstance(p, ast.Call):
            if not y in p.args:
                self._raise('failed to analyze yield expression')
        elif isinstance(p, ast.BinOp):
            if p.left != y and p.right != y:
                self._raise('failed to analyze yield expression')
        elif isinstance(p, ast.Compare):
            if p.left != y and not y in p.comparators:
                self._raise('failed to analyze yield expression')
        else:
            # astpretty.pprint(p)
            if p.value != y:
                self._raise('failed to analyze yield expression')

        if isinstance(p, ast.Expr):
            self.has_terminal_yield = True
        else:
            self.samples = True
            if not isinstance(y.value, ast.Name) or not y.value.id in self.extra_args:
                self._raise('gfuncs can only sample from their arguments')


class GCodeAnalyzer(pysa.VariableAccesses):
    def __init__(self, grammar, code):
        self.grammar = grammar
        self.allowed_ids = set(GFuncAnalyzer.allowed_globals) | set(self.grammar.allowed_global_ids) | set(
            self.grammar.param_ids)
        code.meta = GExprMetadata.DEFAULT.copy()
        self.code = code
        code_ast = ast.parse(code.expr).body[0]

        self.run(code_ast)

    def _raise(self, msg):
        raise GrammaAnalysisException(
            '''in gcode %s parsed by class %s: %s''' % (self.code, self.grammar.__class__.__name__, msg))

    def defs(self, n, v):
        self._raise('gcode cannot modify state')

    mods = defs

    def uses(self, n):
        for p in n.prefixes:
            if p.s in self.allowed_ids:
                return

        for s in self.grammar.reset_states:
            if n.begins(s):
                self.code.meta.statevar_uses.add(n[0].s)
                break
        else:
            self._raise(
                '%s used without being initialized in any reset_state '
                'or explicitly allowed in allowed_global_ids or param_ids' % n.s)


class CacheConfig(object):
    __slots__ = 'randcache', 'statecache'

    def __init__(self):
        self.randcache = {}
        self.statecache = {}

    def new_state(self, val):
        n = '_s%d' % len(self.statecache)
        self.statecache[n] = val
        return n

    def new_randstate(self, val):
        n = '_r%d' % len(self.randcache)
        self.randcache[n] = val
        return n


class GrammaState(object):
    """
    volatile state for sampling, must be initialized in grammar's reset_state.
    """

    def __init__(self):
        self._cache = {}

    def save(self, n, slot):
        self._cache[slot] = copy.deepcopy(getattr(self, n))

    def load(self, n, slot):
        setattr(self, n, copy.deepcopy(self._cache[slot]))


class GrammaRandom(object):
    """
    randomness and stats object.
    """

    __slots__ = 'r', '_cache'

    def __init__(self, seed=None):
        self.r = np.random.RandomState(seed)
        self._cache = {}

    def seed(self, v):
        self.r.seed(v)

    def set_cached_state(self, slot, val):
        self._cache[slot] = val

    def load(self, slot):
        """
        set this random number generator state to the cached value 'slot'.
        """
        st = self._cache.get(slot)
        self.r.set_state(st)

    def save(self, slot):
        """
        store the current random number generator state to 'slot'.
        """
        self._cache[slot] = self.r.get_state()

    def get_state(self):
        return self.r.get_state()

    def set_state(self, st):
        self.r.set_state(st)

    def choice(self, l):
        return l[self.r.randint(0, len(l))]

    def pchoice(self, l, p):
        return self.r.choice(l, p=p)

    def randint(self, low, high):
        return self.r.randint(low, high)

    def rand(self):
        return self.r.rand()

    def geometric(self, p):
        return self.r.geometric(p)

    def f(self, *l, **kw):
        print(l, kw)


class SamplerInterface(object):
    """
    constructed by GrammaSampler and passed to generators for access to random, state, params, and vars.
    """

    __slots__ = 'random', 'state', 'params', 'vars'

    def __init__(self, sampler):
        self.random, self.state, self.params = sampler.random, sampler.state, sampler.params
        self.vars = DictStack()

    def push_vars(self, new_vars):
        self.vars = DictStack(self.vars)
        self.vars.update(new_vars)

    def pop_vars(self):
        self.vars = self.vars.parent


class GrammaSampler(object):
    """
    grammars provide grules, gfuncs, and the reset_state for its gfuncs.

    samplers evaluate GExprs in a stack machine.
    """

    __slots__ = 'grammar', 'transformers', 'sideeffects', 'state', 'random', 'params'

    def __init__(self, grammar=None, **params):
        self.grammar = grammar
        self.transformers = []
        self.sideeffects = []
        self.random = GrammaRandom()
        self.state = GrammaState()
        self.params = type('Params', (), {})
        self.update_params(**params)
        self.add_sideeffects(*self.grammar.sideeffects)

    def add_sideeffects(self, *sideeffects):
        for sideeffect in sideeffects:
            if inspect.isclass(sideeffect):
                sideeffect = sideeffect()
            self.sideeffects.append(sideeffect)

    def add_transformers(self, *transformers):
        for transformer in transformers:
            if inspect.isclass(transformer):
                transformer = transformer()
            self.transformers.append(transformer)

    def update_statecache(self, **kw):
        """keywords are of the form slot=value."""
        self.state._cache.update(kw)

    def get_statecache(self):
        return self.state._cache

    def update_cache(self, cachecfg):
        self.state._cache.update(cachecfg.statecache)
        self.random._cache.update(cachecfg.randcache)

    def update_params(self, **kw):
        for k, v in kw.items():
            setattr(self.params, k, v)

    def sample(self, ge=None):
        return next(self.gensamples(ge))

    def gensamples(self, ge=None):
        if ge is None:
            ge = self.grammar.ruledefs['start']

        if isinstance(ge, string_types):
            ge = self.grammar.parse(ge)

        # do dot operations for loop once
        transformers = self.transformers
        transforms = [transformer.transform for transformer in transformers]
        sideeffects = self.sideeffects
        sideeffect_pushes = [sideeffect.push for sideeffect in sideeffects]
        sideeffect_pops = [sideeffect.pop for sideeffect in sideeffects]
        grammar_compile = self.grammar.compile

        # construct state
        x = SamplerInterface(self)
        stack = deque()

        stack_append = stack.append
        stack_pop = stack.pop

        while True:
            # reset state
            self.grammar.reset_state(x.state)
            for sideeffect in sideeffects:
                sideeffect.reset_state(x.state)

            a = ge

            stillgoing = True
            while stillgoing:
                # assert(isinstance(a,GExpr))

                for transform in transforms:
                    a = transform(x, a)

                sideeffect_top = [push(x, a) for push in sideeffect_pushes]
                compiled_top = grammar_compile(a)(x)
                # wrapped top
                stack_append((sideeffect_top, compiled_top))

                a = next(compiled_top)

                # while isinstance(a,string_types):
                while a.__class__ == str:
                    # pop
                    for pop, w in zip(sideeffect_pops, sideeffect_top):
                        pop(x, w, a)

                    stack_pop()

                    if len(stack) == 0:
                        yield a
                        stillgoing = False
                        break

                    sideeffect_top, compiled_top = stack[-1]
                    a = compiled_top.send(a)

# vim: ts=4 sw=4

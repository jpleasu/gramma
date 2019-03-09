#!/usr/bin/env python
r'''
    Overview
    ========

    Expressions in Gramma are probabilistc programs with string value.  They
    are written in GLF, an extensible syntax for formal language description
    that resembles Backus-Naur form (BNF).

    GLF is extended with custom functions implemented in extensions of the base
    GrammaGrammar class.

    A typical application of Gramma in fuzzing would be as follows:

        Create a grammar based on the input grammar for the application under
        test.

        Feed the instrumented application samples and compute a measure of
        interest for each.

        Tweak numerical parameters of the grammar and/or use previous samples
        as templates to update the grammar.

        Repeat.


    GLF Syntax
    ==========

    - literals - same syntax as Pytnon strings
        'this is a string'
        """this is a (possibly 
            multiline) string"""
    - weighted alternation (|) - weighted random choice from alternatives
        2 x | 3.2 y
        - omitted weights are implicitly 1, so omitting all weights corresponds
          to flat random choice
            x | y

        - weights can also be code in backticks. For example:
            recurs := `depth<5` recurs | "token";
            - this sets the recurse branch weight to 1 (int(True)) if depth <5,
              and 0 otherwise.

    - concatenation (.) - definite concatenation
        x . y
    - repetition ({}) - random repeats
        x{3}
            - generate x exactly 3 times
        x{1,3}
            - generate a number n uniformly in [1,3] then generate x n times
        x{geom(3)}
            - sample a number n from a geometric distribution with mean 3, then
              generate x n times
        x{1,5,geom(3)}
            - same as above, but reject n outside of the interval [1,5]
    - function call (gfuncs) - as defined in a GrammaGrammar extension
        f(x)

        - by inheriting the GrammaGrammar class and adding decorated functions,
          the syntax can be extended.  See below.
        - functions can be stateful, meaning they rely on information stored in
          the GeneratorInterface object.
        - evaluation is left to right.
        - functions aren't allowed to "look up" the execution stack, only back.

    Numerical parameters
    ====================

    Templates from previous samples
    ===============================


    TODO:
        - quirks
            - child_containing treats gfuncs as atomic
            - dynamic alternations use state.depth
                - we could move state.depth into an sideeffect

        - add Transformer class to modify gexprs before they're compiled.
            - a "dynamic" transform can apply to GExprs just before they're
              compiled.. a "static" transformer would transform an entire
              grammar into another grammar, more like inheritance.
            - transformers compose, unlike sideffects. mixing the two is
              too complicated.. sideeffects can both respond to and influence
              the behavior of a transformed grammar.

            - implement LeftAltSideEffect as a transormer and sideffect, e.g.
              replace each alt (in every rule) with a dynamic alt that uses an
              "altdepth" state.

        - compute rdepth, rule depth / recursion depth, for dynamic alt

        - resample should insert the correct load(state,slot) and
          load_rand(slot) in sequence to produce correct output.
          - during tracetree construction, store instate and outstate maps
          - simplify (or "compile") can then remove unnecessary loads.

        - compile an expression -- in preorder traversal, find then
          follow "load_rand" with "definitizing" until next "reseed_rand".
          - we can't necessarily do this in a sideeffect, because we can't
            definitize a stateful node until we know there are no future
            gfuncs accessing a disjoint random stream.

        - implement TreeConstrainedTransformer
            - constructed with expression
                - declarative syntax for finding a node in a tracetree
                - e.g.
                    alt(alt(0),b) - meaning if the left branch of the first alt
                    is taken, take the first option.
                - more astmatcher?

            - add parms.. readonly data usable by gfuncs, no checked initialization

        - decorator to convert function into generator
            - gfunc should be gcoroutine, with a new gfunc decorator that makes
              coroutines of functions.

        - resampling and compilation

            - "resampling" starts with a definitized GExpr one of its tracetree
              nodes.  For example, given the GExpr

                    a(b(),c(),d())

                To resample at "c" we compute:

                    save_rand('r0').a(b(),c().save_rand('r1'),d())

                and store r0 and r1.  The random number generator on entering
                "c" is then reseeded on entry, and resumed with at r1 after
                exiting "c".

                    load_rand('r0').a(b(), reseed_rand().c().load_rand('r1'), d())

                we could also choose the reseed explicitly via a load, e.g. if
                we'd saved a random state "r2" we could use:

                    load_rand('r0').a(b(), load_rand(r2).c().load_rand('r1'), d())

            - the presentation is slightly complicated by recursion. We must
              "unroll" rules until the point where the resampled node occurs.
              e.g. the following generates arrows, "---->"  with length (minus
              1) distributed geometrically.

                    r:= "-".r | ">";

                To resample the "r" node that's three deep in the trace tree of
                "----->", we partially unroll the expression "r":

                    "-".("-".("-".r | ">") | ">") | ">";
                    
                and instrument:

                    save_rand('r0').("-".("-".("-"
                        .r.save_rand('r1') | ">") | ">") | ">");
                    
                then replay with a reseed

                    load_rand('r0').("-".("-".("-"
                        .reseed_rand().r.load_rand('r1') | ">") | ">") | ">");

            - programattically interacting with a tracetree is simpler;
              "load_rand" and "save_rand" are primarily for exposition.

            - "tree order": child < parent

            - "statespace N order": node1 < node2 if both nodes use the same
              statespace, N, and node1 occurs before node2 in preorder
              traversal of the tracetree.
                - the compile GExpr visitation is preorder trace tree traversal,
                  collect a list of nodes for each statespace and random

            - "randstate order": for replay analysis, x.random is treated as a
              statespace used by range, alt, rep, and any gfunc that
              uses_random.  Every use is really a use/def, since the random
              stream changes.

            - the union quasiorder includes tree and state orders, but not
              randstate.

            - for each ordering, if n1<n2 then the value of n1 affects the
              value of n2.  To determine unaffected nodes, we take the
              complement of the upset of the changed node(s) in the union
              quasiorder. every other node will need to be recomputed.


            - where state (including randstate) can be replayed, we can split
              intervals

                - resume by sequencing:

                    def(N, value).f(x)

            - if all descendents of an interval are fixed, can replay each
              node of interval as string

            - if a descendent of a gfunc is changed, must replay the entire
              principal upset to set state e.g. the argument "x" of "g" :
                  
                  f()  ...  g(x)

        - rule-scoped / stacked state spaces?
            - must define stack_reset to initialize top of every stack.
            - to use it, e.g.
                @gfunc
                def f(x):
                    x.state.stacked.beef=7
            - actually.. it makes more sense to have a gfunc that controls the
              stack, e.g. in glf
                r := scoped(def() . "blah". (r . "blah" . use()){2,10} );
            with 
                @gfunc
                def scoped(x):
                    x.state.stack.push
                    x.state.stack_reset
                    res = yield(child)
                    x.state.stack.pop
                    yield res

'''
from __future__ import absolute_import, division, print_function
import sys
if sys.version_info < (3,0):
    #from builtins import (bytes, str, open, super, range,zip, round, input, int, pow, object)
    
    # builtins' object fucks up slots
    from builtins import (bytes, str, open, super, range,zip, round, input, int, pow)
    # builtins str also fucks up isinstance(x,str).. use six.string_types

    import __builtin__

    def func_name(f):
        return f.func_name

    def ast_argname(a):
        return a.id

else:
    import builtins as __builtin__

    def func_name(f):
        return f.__name__

    xrange=range

    def ast_argname(a):
        return a.arg

import copy

#sys.setrecursionlimit(20000)
sys.setrecursionlimit(200000)

from six import string_types, with_metaclass

import lark

import numpy as np

import inspect,ast

from itertools import islice,groupby

from collections import namedtuple

from functools import wraps

import textwrap

import numbers

try:
    import astpretty
except ImportError:
    pass


class GFuncWrap(object):
    __slots__='f','statevar_defs','statevar_uses','fname','uses_random','samples','noauto'
    def __init__(self,f,fname=None,statevar_defs=set(),statevar_uses=set(),noauto=False):
        self.f=f
        self.fname=fname
        self.statevar_defs=set(statevar_defs)
        self.statevar_uses=set(statevar_uses)
        self.noauto=noauto
        if noauto:
            self.samples=True
            self.uses_random=True

    def __call__(self,*l,**kw):
        return self.f(*l,**kw)

    def __str__(self):
        return 'gfunc %s' % self.fname \
            + (' defs=%s' % ','.join(sorted(self.statevar_defs)) if len(self.statevar_defs)>0 else '')\
            + (' uses=%s' % ','.join(sorted(self.statevar_uses)) if len(self.statevar_uses)>0 else '')

def gfunc(*args,**kw):
    '''
        GrammaGrammar function decorator.

        To extend GLF, annote methods of a GrammaGrammar child class with
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
            *) may access entropy from the GeneratorInterface instance
        
        The fields of the GeneratorInterface state used by a gfunc reprsent its
        "state space".  By tracking non-overlapping state spaces, Gramma can
        optimize certain operations.

        gfunc decorator keyword arguments

            fname = string
                provides a name other than the method name to use in glf

            statespaces = list/set
                manual override for automatically inferred state spaces used by
                this function

            
    '''
    def _decorate(f,**kw):
        if not 'fname' in kw:
            kw['fname']=func_name(f)
        return GFuncWrap(f,**kw)

    if len(args)==0 or not callable(args[0]):
        return lambda f:_decorate(f,*args,**kw)

    f=args[0]
    return _decorate(f,**kw)


class GrammaParseError(Exception):
    pass

gramma_grammar = r"""
    ?start : ruledefs

    ruledefs : (ruledef|COMMENT)+

    ruledef : NAME ":=" alt ";"

    ?alt : weight? cat ("|" weight? cat)*

    ?cat : rep ("." rep)*

    ?rep: atom ( "{" rep_args "}" )?

    rep_args : INT ("," INT)? ("," func)?
            | func

    ?atom : string
         | rule
         | func
         | range
         | "(" alt ")"

    rule : NAME

    func.2 : NAME "(" func_args? ")"


    func_args : func_arg ("," func_arg)*

    ?func_arg : code|alt|INT|FLOAT

    ?weight: number| code

    number: INT|FLOAT

    range : "[" ESCAPED_CHAR  ".." ESCAPED_CHAR "]"

    NAME : /[a-z_][a-z_0-9]*/

    string : ESCAPED_CHAR|STRING|LONG_STRING

    code : /`[^`]*`/

    STRING : /[ubf]?r?("(?!"").*?(?<!\\)(\\\\)*?"|'(?!'').*?(?<!\\)(\\\\)*?')/i
    ESCAPED_CHAR.2 : /'([^\']|\\([\nrt']|x[0-9a-fA-F][0-9a-fA-F]))'/
    LONG_STRING.2: /[ubf]?r?("(?:"").*?(?<!\\)(\\\\)*?"(?:"")|'''.*?(?<!\\)(\\\\)*?''')/is

    COMMENT : /#[^\n]*/

    %import common.WS
    %import common.FLOAT
    %import common.INT

    %ignore COMMENT
    %ignore WS
"""


class GExpr(object):
    '''
        the expression tree for a GLF expression.
    '''
    __slots__='parent',
    def __init__(self):
        self.parent=None

    def get_ancestor(self, cls):
        p=self.parent
        while p!=None:
            if isinstance(p,cls):
                return p
            p=p.parent
        return p

    def writeslotsto(self,d):
        'convenience method for deepcopy'
        for a in self.__slots__:
            setattr(d,a,getattr(self,a))

    def is_rule(self, rname=None):
        return False
    
    # tag2cls[lark_tree_node.data]=GExpr_with_parse_larktree_method
    tag2cls={}

    @classmethod
    def parse_larktree(cls,lt):
        if isinstance(lt,lark.lexer.Token):
            return GTok.from_ltok(lt)
        if lt.data=='string':
            return GTok('string',lt.children[0].value)
        if lt.data=='code':
            return GCode(lt.children[0].value[1:-1])

        cls=GExpr.tag2cls.get(lt.data)
        if cls==None:
            raise GrammaParseError('''unrecognized Lark node %s during parse of glf''' % lt)
        return cls.parse_larktree(lt)
 

    def copy(self):
        return None

    def simplify(self):
        'copy self.. caller must ultimately set parent attribute'
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
    __slots__='type','value'
    def __init__(self,type,value):
        GExpr.__init__(self)
        self.type=type
        self.value=value

    def copy(self):
        return GTok(self.type,self.value)

    def __str__(self,children=None):
        return self.value

    def as_native(self):
        if self.type=='INT':
            return self.as_int()
        elif self.type=='FLOAT':
            return self.as_float()
        elif self.type=='string':
            return self.as_str()
        else:
            raise ValueError('''don't recognize GTok type %s''' % self.type)

    def as_int(self):
        return int(self.value)

    def as_float(self):
        return float(self.value)

    def as_str(self):
        return eval(self.value)

    def as_num(self):
        if self.type==u'INT':
            return int(self.value)
        elif self.type==u'FLOAT':
            return float(self.value)
        else:
            raise GrammaParseError('not a num: %s' % self)

    @staticmethod
    def from_ltok(lt):
        return GTok(lt.type,lt.value)

    @staticmethod
    def from_str(s):
        return GTok('string',repr(s))

    @staticmethod
    def join(tok_iter):
        return GTok.from_str(''.join(t.as_str() for t in tok_iter))

    @staticmethod
    def new_empty():
        return GTok.as_str('')

class GInternal(GExpr):
    '''
        nodes with GExpr children
    '''

    # internal nodes must have a tag, corresponding to the larktree data field

    __slots__='children',
    def __init__(self, children):
        GExpr.__init__(self)
        self.children=children
        for c in self.children:
            c.parent=self

    def __str__(self,children=None):
        return '%s(%s)' %(self.__class__.tag, ','.join(str(clt) for clt in children or self.children))

    @classmethod
    def parse_larktree(cls,lt):
        return cls([GExpr.parse_larktree(clt) for clt in lt.children])

    def flat_simple_children(self):
        cls=self.__class__
        children=[]
        for c in self.children:
            c=c.simplify()
            if isinstance(c,cls):
                children.extend(c.children)
            else:
                children.append(c)
        return children

class GCode(object):
    '''
       code expression, e.g. for dynamic alternations
    '''
    def __init__(self, expr):
        self.expr=expr

    def __call__(self, state):
        return eval(self.expr, globals(), {k: getattr(state, k) for k in dir(state)})

    def __str__(self):
        return '`%s`' % self.expr

    def __add__(self,other):
        if isinstance(other,numbers.Number):
            return GCode('(%s)+%f' % (self.expr, other))
        return GCode('(%s)+(%s)' % (self.expr, other.expr))

class GAlt(GInternal):
    tag='alt'

    __slots__='weights','dynamic'
    def __init__(self, weights, children):
        GInternal.__init__(self,children)
        self.dynamic=any(w for w in weights if callable(w))

        if self.dynamic:
            self.weights=weights
        else:
            w=np.array(weights)
            self.weights=w/w.sum()

    def compute_weights(self,state):
        '''
            dynamic weights are computed using the state variable every time an
            alternation is invoked.
        '''
        if self.dynamic:
            w=np.array([w(state) if callable(w) else w for w in self.weights])
            return w/w.sum()
        return self.weights
 
    def __str__(self,children=None):
        #s='|'.join(str(cge) for cge in children or self.children)
        s='|'.join('%s %s' % ('`%s`' % w.expr if callable(w) else w,c) for w,c in zip(self.weights, self.children))

        if self.parent!=None and isinstance(self.parent, (GCat, GRep)):
            return '(%s)' % s
        return s

    def simplify(self):
        if self.dynamic():
            return self.simplify_dynamic()
        return self.simplify_nondynamic()

    def simplify_dynamic(self):
        '''
        complicated normalizing factors could make the result less simple..

            `f1` (`g1` a | `g2` b) | `f2` (`g3` c | `g4` d)

            `f1*g1/(g1+g2)` a | `f1*g2/(g1+g2)` b | `f2*g3/(g3+g4)` c | `f2*g4/(g3+g4)` d

        '''
        return self.copy()

    def simplify_nondynamic(self):
        weights=[]
        children=[]

        for w,c in zip(self.weights, self.children):
            c=c.simplify()
            if isinstance(c,GAlt) and not c.dynamic:
                t=sum(c.weights)
                weights.extend([float(w*cw)/t for cw in c.weights])
                children.extend(c.children)
            else:
                weights.append(w)
                children.append(c)
        if len(children)==0:
            return GTok.new_empty()
        if len(children)==1:
            return children[0]

        # dedupe (and sort) by string representation
        nchildren=[]
        nweights=[]
        for sc, tups in groupby(sorted( (str(c), c, w) for w,c in zip(weights, children) ), key=lambda tup:tup[0]):
            tups=list(tups)
            nweights.append(sum(tup[2] for tup in tups))
            nchildren.append(tups[0][1])

        if len(nchildren)==0:
            return GTok.new_empty()
        if len(nchildren)==1:
            return nchildren[0]
        return GAlt(nweights,nchildren)

    @classmethod
    def parse_larktree(cls,lt):
        weights=[]
        children=[]
        for clt in lt.children:
            if clt.data=='number':
                weights.append(GTok.from_ltok(clt.children[0]).as_num())
                continue
            if clt.data=='code':
                weights.append(GCode(clt.children[0][1:-1]))
                continue
            if len(weights)<=len(children):
                weights.append(1)
            children.append(GExpr.parse_larktree(clt))
        return cls(weights,children)

    def copy(self):
        return GAlt(self.weights, [c.copy() for c in self.children])
 
class GCat(GInternal):
    tag='cat'

    def __str__(self,children=None):
        s='.'.join(str(cge) for cge in children or self.children)
        if self.parent!=None and isinstance(self.parent, GRep):
            return '(%s)' % s
        return s

    def copy(self):
        return GCat([c.copy() for c in self.children])

    def simplify(self):
        children=self.flat_simple_children()
        if len(children)==0:
            return GTok.new_empty()
        if len(children)==1:
            return children[0]

        l=[]
        for t,cl in groupby(children,lambda c:isinstance(c,GTok)):
            if t:
                l.append(GTok.join(cl))
            else:
                l.extend(cl)
        if len(children)==1:
            return children[0]
        return GCat(l)

class GRep(GInternal):
    tag='rep'

    __slots__='rgen', 'lo', 'hi', 'dist'

    def __init__(self,children,lo,hi,rgen,dist):
        GInternal.__init__(self,children)
        self.lo=lo
        self.hi=hi
        self.rgen=rgen
        self.dist=dist

    @property
    def child(self):
        return self.children[0]

    def copy(self):
        return GRep([self.child.copy()],self.lo,self.hi,self.rgen,self.dist)

    def simplify(self):
        return GRep([self.child.simplify()],self.lo,self.hi,self.rgen,self.dist)

    def __str__(self,children=None):
        child=children[0] if children else self.child
        if self.dist=='unif':
            return '%s{%d,%d}' % (child, self.lo,self.hi)
        return '%s{%d,%d,%s}' % (child, self.lo,self.hi,self.dist)

    @classmethod
    def parse_larktree(cls,lt):
        child=GExpr.parse_larktree(lt.children[0])
        args=[GExpr.parse_larktree(c) for c in lt.children[1].children[:]]
        a=args[-1]
        if (not isinstance(a,GTok)) and isinstance(a,GFunc):
            dist=str(a)
            fname=a.fname
            fargs=[x.as_num() for x in a.fargs]
            if fname==u'geom':
                # "a"{geom(n)} has an average of n copies of "a"
                parm=1/float(fargs[0]+1)
                g=lambda x: x.random.geometric(parm)-1
            elif fname=='norm':
                g=lambda x:int(x.random.normal(*fargs)+.5)
            elif fname=='binom':
                g=lambda x:x.random.binomial(*fargs)
            elif fname=='choose':
                g=lambda x:x.random.choice(fargs)
            else:
                raise GrammaParseError('no dist %s' % (fname))

            f=lambda lo,hi:lambda x:min(hi,max(lo,g(x)))
            args.pop()
        else:
            dist='unif'
            f=lambda lo,hi:lambda x:x.random.randint(lo,hi+1)

        if len(args)==0:
            lo=0
            hi=2**32
        elif len(args)==1:
            lo=hi=args.pop(0).as_int()
        else:
            lo=args.pop(0).as_int()
            hi=args.pop(0).as_int()

        rgen=f(lo,hi)
        #print('lo=%d hi=%d' % (lo,hi))
        return GRep([child],lo,hi,rgen,dist)

class GRange(GExpr):
    tag='range'

    __slots__='lo','hi'
    def __init__(self,lo,hi):
        GExpr.__init__(self)
        self.lo=lo
        self.hi=hi

    def copy(self):
        return GRange(self.lo,self.hi)

    def simplify(self):
        if self.hi-self.lo==1:
            return GTok.from_str(chr(self.lo))
        return self.copy()

    def __str__(self,children=None):
        return "['%s' .. '%s']" % (chr(self.lo), chr(self.hi))

    @classmethod
    def parse_larktree(cls,lt):
        lo=ord(GExpr.parse_larktree(lt.children[0]).as_str())
        hi=ord(GExpr.parse_larktree(lt.children[1]).as_str())
        return GRange(lo,hi)

class GFunc(GInternal):
    tag='func'

    __slots__='fname',
    def __init__(self, fname, fargs):
        GInternal.__init__(self,fargs)
        self.fname=fname

    def copy(self):
        return GFunc(self.fname,[c.copy() for c in self.fargs])

    def simplify(self):
        return GFunc(self.fname,[c.simplify() for c in self.fargs])

    @property
    def fargs(self):
        return self.children

    def __str__(self,children=None):
        return '%s(%s)' % (self.fname, ','.join(str(a) for a in children or self.fargs))

    @classmethod
    def parse_larktree(cls,lt):
        fname=lt.children[0].value

        if len(lt.children)>1:
            fargs=[GExpr.parse_larktree(c) for c in lt.children[1].children]
        else:
            fargs=[]

        return GFunc(fname,fargs)

class GRule(GExpr):
    '''
        this is a _reference_ to a rule.. the rule definition is part of the
        GrammaGrammar class
    '''

    tag='rule'

    __slots__='rname',
    def __init__(self,rname):
        GExpr.__init__(self)
        self.rname=rname

    def copy(self):
        return GRule(self.rname)

    def is_rule(self, rname=None):
        if rname==None:
            return True
        return self.rname==rname

    def __str__(self,children=None):
        return self.rname

    @classmethod
    def parse_larktree(cls,lt):
        return GRule(lt.children[0].value)

for cls in GAlt, GCat, GRep, GFunc,   GRange, GRule:
    GExpr.tag2cls[cls.tag]=cls

class GrammaState(object):
    def __init__(self):
        self._cache={}

    def save(self,n,slot):
        self._cache[slot]=copy.deepcopy(getattr(self,n))

    def load(self,n,slot):
        setattr(self,n,copy.deepcopy(self._cache[slot]))

class GrammaRandom(object):

    __slots__='r','_cache'
    def __init__(self,seed=None):
        self.r=np.random.RandomState(seed)
        self._cache={}

    def seed(self,v):
        self.r.seed(v)

    def set_cached_state(self,n,val):
        self._cache[n]=val

    def load(self,n):
        '''
            set this random number generator state to the cached value 'n'
        '''
        st=self._cache.get(n)
        self.r.set_state(st)

    def save(self,n):
        '''
            store the current random number generator state to 'n'
        '''
        self._cache[n]=self.r.get_state()


    def choice(self,l,p=None):
        return self.r.choice(l,p=p)

    def randint(self,low,high):
        return self.r.randint(low,high)

    def geometric(self,p):
        return self.r.geometric(p)

    def f(self,*l,**kw):
        print(l,kw)


class GeneratorInterface(namedtuple('GeneratorInterface','random state parms')):
    '''
        constructed by GrammaSampler and passed to generators for access to
        random and state.
    '''

    def __new__(cls,sampler):
        return super(GeneratorInterface,cls).__new__(cls,sampler.random,sampler.state,sampler.parms)

class GrammaSampler(object):
    '''

        grammars provide grules, gfuncs, and the reset_state for its gfuncs.

        samplers mediate the GExpr requests and responses.

        the context manages state and executes the stack machine to generate
        samples.

    '''
    __slots__='grammar','sideeffects', 'state','random','stack','x','parms'
    def __init__(self,grammar=None, **parms):
        self.grammar=grammar
        self.sideeffects=[]
        self.random=GrammaRandom()
        self.state=GrammaState()
        self.parms=dict(parms)

    def add_sideeffect(self,sideeffect):
        if inspect.isclass(sideeffect):
            sideeffect=sideeffect()
        self.sideeffects.append(sideeffect)

    def update_cache(self, cachecfg):
        self.state._cache.update(cachecfg.statecache)
        self.random._cache.update(cachecfg.randcache)

    def update_parms(self, **kw):
        self.parms.update(kw)

    def reset(self):
        self.state.depth=0

        self.grammar.reset_state(self.state)

        for sideeffect in self.sideeffects:
            sideeffect.reset(self.state)

        self.x=GeneratorInterface(self)
        self.stack=[]

    def sample(self,ge=None):
        if ge==None:
            #ge=self.grammar.ruledefs['start']
            ge='start'

        if isinstance(ge,string_types):
            ge=self.grammar.parse(ge)

        self.reset()

        a=ge
        wtops=[None]*len(self.sideeffects)
        while True:
            #assert(isinstance(a,GExpr))

            #push
            setop=tuple(sideeffect.push(self.x,a) for sideeffect in self.sideeffects)
            top=self.grammar.compile(a)(self.x)
            wtop=(setop,top) 

            self.state.depth+=1
            self.stack.append(wtop)

            a=next(top)
            while isinstance(a,string_types):
                #pop
                for sideeffect,w in zip(self.sideeffects,wtop[0]):
                    sideeffect.pop(self.x,w,a)

                self.stack.pop()
                self.state.depth-=1

                if len(self.stack)==0:
                    return a

                wtop=self.stack[-1]
                a=wtop[1].send(a)


class NullSideEffect(object):
    '''
        An execution sideeffect observes key points during sampler execution.
    '''
    __slots__=()
    def reset(self,state):
        '''
        called right after sample, before the stack machine starts
        '''
        pass

    def push(self,x,ge):
        '''
        when an expression is compiled to a coroutine, the stack machine
        pushes.
        '''
        return None

    def pop(self,x,w,s):
        '''
        when a coroutine complete, returning a string, s, it is popped.
        w is the value returned by the corresponding pop.
        '''
        pass

class GrammaGrammarException(Exception):
    pass

def ast_attr_path(x):
    p=[x]
    while isinstance(p[0].value,ast.Attribute):
        p.insert(0,p[0].value)
    return p


class GFuncAnalyzeVisitor(ast.NodeVisitor):
    '''
        A python AST node visitor for @gfunc decorated methods of a
        GrammaGrammar child class.

        This is used by analyze_gfuncs to populate GFuncWrap objects with state
        spaces, don't use directly.
    '''
    allowed_globals=['struct','True','False','None'] + [x for x in dir(__builtin__) if x.islower()]

    def __init__(self,target_class,f,allowed_ids=None):
        self.target_class=target_class
        # id of the GeneratorInterface (first argument of gfunc)
        # other argument ids
        self.allowed_ids=set(GFuncAnalyzeVisitor.allowed_globals)
        if allowed_ids!=None:
            self.allowed_ids.update(allowed_ids)
        self.uses_random=False
        self.samples=False
        self.has_terminal_yield=False

        self.statevar_defs=set()
        self.statevar_uses=set()

        self.f=f
        al=f.args.args
        self.iface_id=ast_argname(al[0])
        # XXX forbid args with default values?
        self.allowed_ids.update(ast_argname(a) for a in al)

        self.stack=[]

    def visit(self,n):
        self.stack.append(n)
        super().visit(n)
        self.stack.pop()

    def run(self):
        for item in self.f.body:
            self.visit(item)

    def is_iface_id(self,n):
        if isinstance(n,ast.Name) and n.id==self.iface_id:
            return True
        return isinstance(n,ast.Attribute) and self.is_iface_id(n.value)


    def call_attrs_subsr(self):
        'test for a method call a subscript object'
        if isinstance(self.stack[-1],ast.Subscript):
            try:
                n=next(n for n in reversed(self.stack[:-1]) if not isinstance(n,ast.Attribute))
                if isinstance(n,ast.Call):
                    return True
            except StopIteration:
                pass
        return False

    def visit_iface_id(self,x,attrs=None,in_subscript=False):
        if isinstance(x,ast.Name):
            nm=x.id
            if attrs!=None:
                nm+=attrs
            raise GrammaGrammarException('Forbidden access (%s) on line %d!!' % (nm, x.lineno))

        elif isinstance(x,ast.Attribute):
            p=ast_attr_path(x)
            vname='.'.join(x.attr for x in p)
            if len(p)<=1:
                raise GrammaGrammarException('Forbidden access to %s, access field by name.' % vname)
            attr=p[0].attr
            if attr=='random':
                self.uses_random=True
            elif attr=='state':
                d,u=False,False
                par=self.stack[-1]

                #print(vname)
                #print(self.stack)
                #astpretty.pprint(self.stack[-1])

                # if len(p)>3 and p[2]=='stacked': ...

                if len(p)>2 or isinstance(par, ast.AugAssign) or self.call_attrs_subsr():
                    # need to discriminate for dynamic statespaces
                    d=u=True
                elif isinstance(par,ast.Assign):
                    d=True
                else:
                    u=True
                if d:
                    self.statevar_defs.add(p[1].attr)
                if u:
                    self.statevar_uses.add(p[1].attr)
            else:
                raise GrammaGrammarException('''forbidden use of %s.%s in gfunc %s of class %s on line %d''' % (self.iface_id, attr, self.f.name, self.target_class.__name__, x.lineno))

        else:
            raise GrammaGrammarException('iface_id not Attribute or Name? %s' % (x))

    def visit_Subscript(self, sub):
        if self.is_iface_id(sub.value):
            self.visit_iface_id(sub.value,in_subscript=True)
        else:
            self.visit(sub.value)
        self.visit(sub.slice)

    def visit_Yield(self, y):
        self.visit(y.value)

        if any(n for n in self.stack[:-1] if isinstance(n,(ast.GeneratorExp,ast.ListComp))):
            raise GrammaGrammarException("yield in a generator expression or list comprehension, in gfunc %s of class %s on line %d" % (self.f.name, self.target_class.__name__, y.lineno))

        p=self.stack[-2]
        if p.value!=y:
            raise GrammaGrammarException('failed to analyze yield expression in gfunc %s of class %s on line %d' % (self.f.name, self.target_class.__name__, y.lineno))

        if isinstance(p,ast.Expr):
            self.has_terminal_yield=True
        else:
            self.samples=True

    def visit_AugAssign(self, ass):
        self.visit(ass.value)
        if self.is_iface_id(ass.target):
            self.visit_iface_id(ass.target)
        elif isinstance(ass.target,ast.Subscript):
            if self.is_iface_id(ass.target.value):
                self.visit_iface_id(ass.target.value,in_subscript=True)

    def visit_Assign(self, ass):
        self.visit(ass.value)
        for a in ass.targets:
            if self.is_iface_id(a):
                self.visit_iface_id(a)
            else:
                if isinstance(a,ast.Subscript):
                    if self.is_iface_id(a.value):
                        self.visit_iface_id(a.value,in_subscript=True)
                        return
                self.allowed_ids.add(a.id)

    def visit_Attribute(self,a):
        if self.is_iface_id(a):
            self.visit_iface_id(a)
        else:
            self.visit(a.value)

    def visit_Name(self,n):
        if n.id==self.iface_id:
            self.visit_iface_id(n)
        elif not n.id in self.allowed_ids:
            raise GrammaGrammarException('''forbidden use of variable '%s' in %s on line %d
            of %s! to explicitly allow append to GrammaGrammar class variable
            list, ALLOWED_GLOBAL_IDS. To prevent analysis of this function, add
            @gfunc(noauto=True) ''' % (n.id, self.f.name, n.lineno, self.target_class.__name__))

        # done
    def visit_FunctionDef(self,f):
        # don't descend!
        pass

class ResetStateAnalyzer(ast.NodeVisitor):
    '''
        A python AST node visitor for the reset_state method of GrammaGrammar
        child classes.

        This is used by analyze_gfuncs to check that all state spaces are
        initialized in reset_state.

        XXX: There isn't a way for a user to promise that some state space has
        been initialized without assigning to it in the reset_state function.
        More decorators?

    '''
    def __init__(self,target_class,statespace_usedby,f):
        self.target_class=target_class
        self.statespace_usedby=statespace_usedby
        self.f=f
        al=f.args.args
        self.state_id=ast_argname(al[1])
        self.assigned_statespaces=set()

    def run(self):
        for item in self.f.body:
            self.visit(item)

        throwme=[]
        for uid,fs in self.statespace_usedby.items():
            if not uid in self.assigned_statespaces:
                if len(fs)==1:
                    fss="gfunc '%s'" % next(iter(fs))
                else:
                    fss="gfuncs {%s}" % (','.join("'%s'" % fn for fn in fs))
                throwme.append('''statespace '%s' is used by %s''' %(uid,fss))
        if len(throwme)>0:
            raise GrammaGrammarException('%s: initialize state fields in reset_state method of %s!' % (', '.join(throwme), self.target_class.__name__))

    def visit_Assign(self, ass):
        for a in ass.targets:
            self.visit_state_var(a)
        self.visit(ass.value)

    def visit_state_var(self,n):
        'if n is a state variable reference, add it to the visited set'
        p=ast_attr_path(n)
        n=p[0]
        if isinstance(n,ast.Attribute) and n.value.id==self.state_id:
            self.assigned_statespaces.add(n.attr)

    def visit_FunctionDef(self,f):
        pass

def analyze_gfuncs(GrammaChildClass,allowed_ids=None):
    '''
        Enumerate @gfunc decorated methods of GrammaChildClass in order to
        infer state spaces.

        methods tagged with "auto=False" will be skipped.. you're on your own.

    '''
    s=inspect.getsource(GrammaChildClass)
    s=textwrap.dedent(s)

    classdef=ast.parse(s).body[0]
    def isgfuncdec(y):
        if isinstance(y,ast.Name) and y.id=='gfunc':
            return True
        return isinstance(y,ast.Call) and y.func.id=='gfunc'

    gfuncs=[x for x in classdef.body if isinstance(x,ast.FunctionDef) and any(isgfuncdec(y) for y in x.decorator_list)]
    statespace_usedby={}
    #print('class %s' % GrammaChildClass.__name__)
    for gf_ast in gfuncs:
        gf=getattr(GrammaChildClass,gf_ast.name)
        if gf.noauto:
            continue

        #print('==== %s ====' % gf_ast.name)
        #astpretty.pprint(gf_ast)

        analyzer=GFuncAnalyzeVisitor(GrammaChildClass,gf_ast,allowed_ids)
        analyzer.run()
        if not analyzer.has_terminal_yield:
            raise GrammaGrammarException('''gfunc %s of class %s doesn't yield a value''' % (gf_ast.name, GrammaChildClass.__name__))

        gf.statevar_defs.update(analyzer.statevar_defs)
        gf.statevar_uses.update(analyzer.statevar_uses)
        gf.uses_random=analyzer.uses_random
        gf.samples=analyzer.samples

        for ss in gf.statevar_defs|gf.statevar_uses:
            statespace_usedby.setdefault(ss,set()).add(gf_ast.name)

        #print(gf)

    #print(statespace_usedby)
    reset_state_ast=([x for x in classdef.body if isinstance(x,ast.FunctionDef) and x.name=='reset_state']+[None])[0]
    if len(statespace_usedby.keys())>0 and reset_state_ast==None:
        raise GrammaGrammarException('%s has no reset_state method, but uses statespace(s) %s' % (GrammaChildClass.__name__, ','.join(sorted(statespace_usedby.keys()))))

    if reset_state_ast != None:
        analyzer=ResetStateAnalyzer(GrammaChildClass,statespace_usedby,reset_state_ast)
        analyzer.run()

    

class GrammaGrammarType(type):
    '''
        metaclass to kick off analysis of gfuncs in children of GrammaGrammar
    '''
    #def __new__(metaname, classname, baseclasses, attrs):
    #    return type.__new__(metaname, classname, baseclasses, attrs)

    def __init__(classobject, classname, baseclasses, attrs):
        analyze_gfuncs(classobject, getattr(classobject, 'ALLOWED_GLOBAL_IDS', []))
        #print('done with analysis of %s' % classname)

class GrammaGrammar(with_metaclass(GrammaGrammarType,object)):
    '''
        The class defining functions and state management for a gramma and
        extensions.

        e.g. 
         g=GrammaGrammar('start:="a"|"b";')
         sampler=GrammaSampler(g)

         while True:
             sampler.reset()
             s=sampler.sample()
             print(s)
    '''

    ALLOWED_GLOBAL_IDS=[]

    parser = lark.Lark(gramma_grammar, parser='earley', lexer='standard')

    def __init__(self, gramma_expr_str):
        self.ruledefs={}
        self.funcdefs={}

        for n,f in inspect.getmembers(self,predicate=lambda x:isinstance(x,GFuncWrap)):
            self.funcdefs[f.fname]=f

        lt=self.parser.parse(gramma_expr_str)
        for ruledef in lt.children:
            rname=ruledef.children[0].value
            rvalue=GExpr.parse_larktree(ruledef.children[1])
            self.ruledefs[rname]=rvalue

    def reset_state(self,state):
        state.rlim_depth=0

    @gfunc
    def save_rand(x,slot):
        x.random.save(slot.as_str())
        yield ''

    @gfunc
    def load_rand(x,slot):
        x.random.load(slot.as_str())
        yield ''

    @gfunc
    def reseed_rand(x):
        x.random.seed(None)
        yield ''


    @gfunc(noauto=True)
    def save(x,n,slot):
        x.state.save(n.as_str(),slot.as_str())
        yield ''

    @gfunc(noauto=True)
    def load(x,n,slot):
        x.state.load(n.as_str(),slot.as_str())
        yield ''

    @gfunc(fname='def',noauto=True)
    def def_(x,n,v):
        if isinstance(v,GTok):
            v=v.as_native()
        elif isinstance(v,GCode):
            v=v(x.state)
        setattr(x.state,n.as_str(), v)
        yield ''



    @gfunc
    def rlim(x,c,n,o):
        '''
            recursion limit - if recursively invoked to a depth < n times,
                sample c, else sample o

            e.g. 

                R :=  rlim("a" . R, 3, "");

                produces "aaa" 
            
        '''
        x.state.rlim_depth+=1
        n=n.as_int()
        res=yield (c if x.state.rlim_depth<=n else o)
        x.state.rlim_depth-=1
        yield res

    def parse(self,gramma_expr_str):
        '''
            parse gramma_expr_sr with the current rules and functions and
            return a GExpr object
        '''
        lt=self.parser.parse('_:=(%s);' % gramma_expr_str)
        return GExpr.parse_larktree(lt.children[0].children[1])

    def generate(self,startexpr=None):
        '''
            quick and dirty generator
        '''
        sampler=GrammaSampler(self)
        if startexpr==None:
            startexpr=self.ruledefs['start']
        while True:
            yield sampler.sample(startexpr) 


    def compile(self, ge):
        if isinstance(ge,GTok):
            def g(x):
                yield ge.as_str()
        elif isinstance(ge,GAlt):
            if ge.dynamic:
                def g(x):
                    s=yield x.random.choice(ge.children,p=ge.compute_weights(x.state))
                    yield s
            else:
                def g(x):
                    s=yield x.random.choice(ge.children,p=ge.weights)
                    yield s
        elif isinstance(ge,GCat):
            def g(x):
                s=''
                for cge in ge.children:
                    s+=yield cge
                yield s
        elif isinstance(ge,GRep):
            def g(x):
                s=''
                n=ge.rgen(x)
                while n>0:
                    s+=yield(ge.child)
                    n-=1
                yield s
        elif isinstance(ge, GRange):
            def g(x):
                yield chr(x.random.randint(ge.lo,ge.hi+1))
        elif isinstance(ge,GRule):
            rhs=self.ruledefs[ge.rname]
            def g(x):
                s=yield(rhs)
                yield s
        elif isinstance(ge, GFunc):
            gf=self.funcdefs.get(ge.fname,None)
            if gf==None:
                raise GrammaGrammarException('no gfunc named %s available to %s' % (ge.fname,self.__class__.__name__))
            gargs=ge.fargs
            def g(x):
                return gf(x,*gargs)
        else:
            raise GrammaGrammarException('unrecognized expression: (%s) %s' % (type(ge), ge))
        return g

class CacheConfig(object):
    __slots__='randcache','statecache'


    def __init__(self):
        self.randcache={}
        self.statecache={}

    def new_state(self, val):
        n='_s%d' % len(self.statecache)
        self.statecache[n]=val
        return n

    def new_randstate(self, val):
        n='_r%d' % len(self.randcache)
        self.randcache[n]=val
        return n

class TTNode(object):
    '''
        a node of the tracetree
    '''
    __slots__='ge','parent','children','s','inrand','outrand'
    def __init__(self,ge):
        self.ge=ge
        self.parent=None
        self.children=[]
        self.s=None
        self.inrand=None
        self.outrand=None

    def add_child(self,ge):
        c=TTNode(ge)
        c.parent=self
        self.children.append(c)
        return c

    def dump(self,indent=0,out=sys.stdout):
        print('%s%s -> "%s"' % ('  '*indent, self.ge, self.s),file=out)
        for c in self.children:
            c.dump(indent+1,out)

    def inbounds(self,i,j):
        '[i,j) contained in [0,len(self.s))?'
        return 0<=i and j<=len(self.s)

    def child_containing(self, i,j=None, d=0):
        if j==None or j < i+1:
            j=i+1

        #print('%s[%d,%d) %s' % (' '*d, i,j,self.ge))

        if isinstance(self.ge, (GRule,GAlt)):
            return self.children[0].child_containing(i,j,d+1)

        # don't descend into GFuncs
        if isinstance(self.ge, (GCat, GRep)):
            # i         v
            #   aaaa  bbbb   cccc
            #    
            o=0
            for c in self.children:
                x=c.child_containing(i-o,j-o,d+1)
                if x!=None:
                    return x
                o+=len(c.s)
            if self.inbounds(i,j):
                return self
            return None

        if isinstance(self.ge,(GTok, GFunc, GRange)):
            return self if self.inbounds(i,j) else None

        raise GrammaParseError('unknown expression (%s)%s' % (type(self.ge), self.ge))

    def first_rule(self,rname):
        for n in self.gennodes():
            if n.ge.is_rule(rname):
                return n
        return None

    def gennodes(self):
        yield self
        for c in self.children:
            for cc in c.gennodes():
                yield cc

    def depth(self,pred=lambda x:True):
        '''
            # of ancestors that satisfy pred
        '''
        n=self.parent
        d=0
        while n!=None:
            if pred(n):
                d+=1
            n=n.parent
        return d

    def resample(self,grammar,pred):
        '''
            computes ge, a GExpr that resamples only the nodes satisfying pred.

            also computes cfg, a CacheConfig populated with any extra random and state
            values needed by ge.

            return ge,cfg
        '''

        cachecfg=CacheConfig()

        def recurse(t):
            if pred(t):
                slot=cachecfg.new_randstate(t.outrand)
                return GCat([grammar.parse('reseed_rand()'),t.ge.copy(),grammar.parse("load_rand('"+slot+"')")])

            ge=t.ge
            if isinstance(ge,GAlt):
                # XXX lost sample, rand stream out of sync w/ original
                return recurse(t.children[0])
            elif isinstance(ge,GRule):
                return recurse(t.children[0])
            elif isinstance(ge,GCat):
                return GCat([recurse(c) for c in t.children])
            elif isinstance(ge,GRep):
                # XXX lost sample, rand stream out of sync w/ original
                return GCat([recurse(c) for c in t.children])
            elif isinstance(ge,GRange):
                return GTok.from_str(t.s)
            elif isinstance(ge,GTok):
                return ge.copy()
            elif isinstance(ge,GFunc):
                return ge.copy()
            else:
                raise ValueError('unknown GExpr node type: %s' % type(ge))

        return recurse(self).simplify(), cachecfg

class TracingSideEffect(object):
    __slots__='tt','tracetree'

    def reset(self,state):
        self.tracetree=None

    def push(self,x,ge):
        if self.tracetree==None:
            self.tt=self.tracetree=TTNode(ge)
        else:
            self.tt=self.tt.add_child(ge)
        self.tt.inrand=x.random.r.get_state()
        return None

    def pop(self,x,w,s):
        self.tt.s=s
        self.tt.outrand=x.random.r.get_state()
        self.tt=self.tt.parent

# vim: ts=4 sw=4

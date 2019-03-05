#!/usr/bin/env python
r'''
    Overview
    ========

    Expressions in Gramma are probabilistc programs with string value.  They
    are written in GLF, an extensible syntax for formal language description
    that resembles Backus-Naur form (BNF).

    GLF is extended with custom functions implemented in extensions of the base
    GrammaGrammar class.

    The typical usage of Gramma for fuzzing is as follows:
        A grammar is constructed based on the input grammar for the application
        under test.

        The instrumented application is fed samples, and a measure of interest
        is calculated for each.

        Parameters of the grammar are reestimated to favor the interesting
        samples.

        The process continues with updated grammar.


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

    gfuncs
    ------
    Functions in Gramma are implemented by @gfunc decorated methods of
    GrammaGrammar objects.  Without narrowing the influence of a gfunc, Gramma
    assumes it might use any state, which restricts Gramma's ability to optmize
    sampling from contrained expressions that use it.  For example, consider
    the expression
        f() . ("a" | "b") . f()
    under the constraint
        cat(child1="x", child2=alt(c=2,alt(c=2,"b")))
    Because the first call to 'f' might have side effects that the second call
    relies on, we can't replace it with a constant "x". If, on the other hand,
    we knew that 'f' was stateless, we could compile the constraints expression
    to
        "xa".f()
    PyDoc of the gfunc decorator gives details.




    Constraints
    ===========

    For parameter estimation and resampling conditional distribtuions, Gramma
    provides a mechanism to construct and apply _tree constraints_.

    "Tree constraints" provide a restricted, but useful, filter for the
    evaluation of a Gramma expression. For example, sampling from
        ("a" | "b"){1,2}
    results in "a", "b", "aa", "ab", "ba", "bb", with the diagraphs half as
    likely as the single character strings.

    The execution resulting in "ab" can be depicted with a tree as follows:

        rep
        / \
      alt alt
       |   |
       a   b

    Where the rep and each alt have made random choices.  We might be
    intersested in the other results where some, but not all, of those random
    decisions are made same way.

    For example, we might want the root, rep, to always choose 2 and the second
    alt to choose "b" -- the resulting constrained sampler would produce "ab"
    and "bb" with equal probability.

    Whatever the representation for constraints, we should be able to combine
    them with conjuction (and disjunction?) to be efficient.  Tree regular
    predicate language?

    The _defining_ constraint on ("a"|"b"){1,2} that produced "ab" is
        rep(n=2,child1=alt(c=1,child="a"), child2=alt(c=2,child="b"))
    By omitting the constraint on child1 of the root rep, we get
        rep(n=2,child2=alt(c=2,child="b"))
    In this case, we can compile this constrained expression to
        ("a"|"b")."b"




    TODO:
        - TTNode
            - find the deepest child responsible for a given position
                - function that reverses string would screw up children?? treat
                  all functions as opaque?

        - programatic resample.. generate a tracetree, pick a node, generate a gexpr and resample it
            - compile such an expression -- in preorder traversal, find then
              follow "load_rand" with "definitizing" until next "reseed_rand".
              - we can't necessarily do this in a sampler, because we can't
                definitize a stateful node until we know there are no future
                gfuncs accessing a disjoint random stream.

        - conditioning
            - in general, force the random number generator to return a list of
              values:

                "cond(expr, 1, 2, 3)"

        - implement TreeConstrainedSampler
            - constructed with expression
                - e.g.
                    alt(alt(0),b) - meaning if the left branch of the first alt
                    is taken, take the first option.
                - maybe something horribly xpathy? more astmatcher?
                - ability to randomly construct a tree contraint that selects
                  one of the nodes of a given TraceTree

        - add parms.. readonly data for use by gfuncs

        - compiler
            - save state "def, use" tags for each statespace and for random.
            - convert functions into generators
            - analyze generator
                - comment on nested yields

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

            - "state order N ": node1 < node2 if both nodes use the same
              statespace, N, and node1 occurs before node2 in preorder
              traversal of the tracetree.
                - the expr2ctor GExpr visitation is preorder trace tree traversal,
                  collect a list of nodes for each statespace and random

            - "randstate order": for replay analysis, x.random is treated as a
              statespace used by range, alt, rep, and any gfunc that
              uses_random.

            - the union quasiorder includes tree and state orders, but not
              randstate.

            - for each ordering, if n1<n2 then the value of n1 affects the
              value of n2.  To determine unaffected nodes, we take the
              complement of the upset of the changed node(s) in the union
              quasiorder. every other node will need to be recomputed.


            - where state (including randstate) can be replayed, we can split
              intervals

                - resume by sequencing:

                    set(N, value).f(x)

            - if all descendents of an interval are fixed, can replay each
              node of interval as string

            - if a descendent of a gfunc is changed, must replay the entire
              principal upset to set state e.g. the argument "x" of "g" :
                  
                  f()  ...  g(x)



        - documentation
            - gfunc authoring
            - sampler internals
                - expr2ctor observes preorder GExprs and postorder strings during
                  the (lazy) depth first traversal of of the tracetree.
                - "push exprs pop strings"

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
                    x.sample(child)
                    x.state.stack.pop

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

#sys.setrecursionlimit(20000)
sys.setrecursionlimit(200000)

from six import string_types, with_metaclass

import lark

import numpy as np

import inspect,ast

from itertools import islice,groupby

from collections import namedtuple

from functools import wraps

try:
    import astpretty
except ImportError:
    pass


class GFuncWrap(object):
    __slots__='f','statevars','fname','uses_random','calls_sample','noauto'
    def __init__(self,f,fname=None,statevars=set(),noauto=False):
        self.f=f
        self.fname=fname
        self.statevars=set(statevars)
        self.noauto=noauto
        if noauto:
            self.calls_sample=True
            self.uses_random=True

    def __call__(self,*l,**kw):
        return self.f(*l,**kw)

    def __str__(self):
        return 'gfunc %s statevars=%s' %(self.fname, self.statevars)

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
                        x.random - an alias for x.sampler.random

                    args are the arguments of f as GExpr elements.  In
                    particular, "constants" are are type GTok and must be
                    converted, and generate GEXpr objects can be sampled from.

            *) mustn't access global variables
            *) may store state as fields of the GrammaState instance, state
            *) mustn't take additional keyword arguments, only "grammar",
                "sampler", and "state" are allowed.
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
        return GFuncWrap(f,fname=func_name(f),**kw)

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

    ?alt : number? cat ("|" number? cat)*

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

    ?func_arg : alt|INT|FLOAT

    number: INT|FLOAT

    range : "[" ESCAPED_CHAR  ".." ESCAPED_CHAR "]"

    NAME : /[a-z_][a-z_0-9]*/

    string : ESCAPED_CHAR|STRING|LONG_STRING


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


class GParseError(Exception):
    pass

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

    def isrule(self, rname):
        return False
    
    def is_stateful(self,x,assume_no=None):
        '''
            does sampling have side effects on the sampler other than using x.random?
        '''
        return True

    # tag2cls[lark_tree_node.data]=GExpr_with_parse_larktree_method
    tag2cls={}

    @classmethod
    def parse_larktree(cls,lt):
        if isinstance(lt,lark.lexer.Token):
            return GTok.from_ltok(lt)
        if lt.data=='string':
            return GTok('string',lt.children[0].value)

        cls=GExpr.tag2cls.get(lt.data)
        if cls==None:
            raise GParseError('''unrecognized Lark node %s during parse of glf''' % lt)
        return cls.parse_larktree(lt)
 

    def copy(self):
        return None

    def simplify(self):
        'copy self.. caller must ultimately set parent attribute'
        return self.copy()


    def as_num(self):
        raise GrammaParseError('''only tokens (literal numbers) have an as_num method''')

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

    def as_int(self):
        return int(self.value)

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

    def is_stateful(self,x,assume_no=None):
        return False

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
    
    def copy(self):
        cls=self.__class__
        return cls([c.copy() for c in self.children])

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

class GAlt(GInternal):
    tag='alt'

    __slots__='weights',
    def __init__(self, weights, children):
        GInternal.__init__(self,children)
        t=sum(weights)
        weights=np.array([float(w)/t for w in weights])
        self.weights=weights
 
    def __str__(self,children=None):
        #s='|'.join(str(cge) for cge in children or self.children)
        s='|'.join('%s %s' % (w,c) for w,c in zip(self.weights, self.children))

        if self.parent!=None and isinstance(self.parent, (GCat, GRep)):
            return '(%s)' % s
        return s

    def is_stateful(self,x,assume_no=None):
        return any(c.is_stateful(x,assume_no) for c in self.children)

    def simplify(self):
        weights=[]
        children=[]

        for w,c in zip(self.weights, self.children):
            c=c.simplify()
            if isinstance(c,GAlt):
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
            if len(weights)<=len(children):
                weights.append(1)
            children.append(GExpr.parse_larktree(clt))
        return cls(weights,children)


 
class GCat(GInternal):
    tag='cat'

    def __str__(self,children=None):
        s='.'.join(str(cge) for cge in children or self.children)
        if self.parent!=None and isinstance(self.parent, GRep):
            return '(%s)' % s
        return s

    def is_stateful(self,x,assume_no=None):
        return any(c.is_stateful(x,assume_no) for c in self.children)

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

    def is_stateful(self,x,assume_no=None):
        return self.child.is_stateful(x,assume_no)

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

    def is_stateful(self,x,assume_no=None):
        return False

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

    def is_stateful(self,x,assume_no=None):
        # functions can recurse too
        if assume_no==None:
            assume_no=set()
        if self.fname in assume_no:
            return False
        #XXX function and rule collision
        assume_no.add(self.fname)
        return x.funcdefs[self.fname].stateful or any(c.is_stateful(x,assume_no) for c in self.fargs)

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

    def isrule(self,rname):
        return self.rname==rname

    def __str__(self,children=None):
        return self.rname

    @classmethod
    def parse_larktree(cls,lt):
        return GRule(lt.children[0].value)

    def is_stateful(self,x,assume_no=None):
        if assume_no==None:
            assume_no=set()
        if self.rname in assume_no:
            return False
        assume_no.add(self.rname)
        return x.ruledefs[self.rname].is_stateful(x,assume_no)

for cls in GAlt, GCat, GRep, GFunc,   GRange, GRule:
    GExpr.tag2cls[cls.tag]=cls

class SamplerException(Exception):
    pass

class GrammaState(object):
    pass


class GrammaRandom(object):

    __slots__='r','cache'
    def __init__(self,seed=None):
        self.r=np.random.RandomState(seed)
        self.cache={}

    def seed(self,v):
        self.r.seed(v)

    def set_cached_state(self,n,val):
        self.cache[n]=val

    def get_cached_state(self,n):
        return self.cache[n]

    def load_state(self,n):
        '''
            set this random number generator state to the cached value 'n'
        '''
        st=self.cache.get(n)
        self.r.set_state(st)

    def save_state(self,n):
        '''
            store the current random number generator state to 'n'
        '''
        self.cache[n]=self.r.get_state()


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
        constructed by SamplerContext and passed to generators for access to
        random and state.
    '''

    def __new__(cls,sampler):
        return super(GeneratorInterface,cls).__new__(cls,sampler.random,sampler.state,sampler.parms)

class SamplerContext(object):
    '''
        the grammar provides grules (including a "start" rule), gfuncs, and the
        reset_state function.
    '''
    __slots__='grammar','state','random','stack','x','parms'
    def __init__(self,grammar,**parms):
        self.grammar=grammar
        self.random=GrammaRandom()
        self.state=GrammaState()
        self.parms=dict(parms)

    def update_parms(self, **kw):
        self.parms.update(kw)

    def reset(self):
        self.grammar.reset_state(self.state)
        self.random.save_state('__initial_random')

        self.x=GeneratorInterface(self)
        self.stack=[]

    def sample(self,sampler,ge=None):
        if ge==None:
            ge=self.grammar.ruledefs['start']

        if isinstance(ge,string_types):
            ge=self.grammar.parse(ge)

        self.reset()
        sampler.reset()

        self.stack.append(sampler.expr2ctor(ge)(self.x))
        while True:
            a=next(sampler.unwrap(self.stack[-1]))
            while isinstance(a,string_types):
                sampler.complete(self.stack[-1],a)
                self.stack.pop()
                if len(self.stack)==0:
                    return a
                a=sampler.unwrap(self.stack[-1]).send(a)
            self.stack.append(sampler.expr2ctor(a)(self.x))


class DefaultSampler(object):
    '''
        A Sampler is a tuple of functions used for sampling.
    '''
    __slots__='grammar',
    def __init__(self,grammar):
        self.grammar=grammar

    def reset(self):
        '''
        called before sampling from the top of an expression
        '''
        pass

    def unwrap(self,top):
        '''
        return a generator/coroutine from a stack object
        '''
        return top

    def complete(self,top,s):
        '''
        called when the top coroutin completes, i.e. returns a string.

        top is the stack object, and s is the string it generated.
        '''
        pass

    def expr2ctor(self,ge):
        '''
            Given a GExpr, return a constructor taking a single
            GeneratorInterface argument.
            
            The context stack will be composed of objects constructed with
            these ctors.
            
            Stack objects must behave like coroutines, responding to "next" and
            "send", as in
                g=expr2ctor(gexpr)
                next(g) # returns str or GExpr
                g.send(str) # returns str or GExpr

            "unwrap" can be used for simple wrappers:

                def expr2ctor(self,ge):
                    top=super().expr2ctor(ge)
                    return (top,ge)

                def unwrap(self,top):
                    top,ge=top
                    return super().unwrap(top)

        '''
        if isinstance(ge,GTok):
            def g(x):
                yield ge.as_str()
        elif isinstance(ge,GAlt):
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
            rhs=self.grammar.ruledefs[ge.rname]
            def g(x):
                s=yield(rhs)
                yield s
        elif isinstance(ge, GFunc):
            gf=self.grammar.funcdefs[ge.fname]
            gargs=ge.fargs
            def g(x):
                return gf(x,*gargs)
        else:
            raise SamplerException('unrecognized expression: (%s) %s' % (type(ge), ge))
        return g


class ProxySampler(object):
    '''
        derive from this to avoid duplicating common requirements on a sampler
    '''
    __slots__='base',

    def __init__(self,base):
        self.base=base

    def reset(self):
        self.base.reset()

    def unwrap(self,top):
        return self.base.unwrap(top)

    def complete(self,top,s):
        self.base.complete(top,s)

    def expr2ctor(self,ge):
        return self.base.expr2ctor(ge)

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
        self.calls_sample=False
        self.statevars=set()

        self.f=f
        al=f.args.args
        self.iface_id=ast_argname(al[0])
        # XXX prevent args with default values?
        self.allowed_ids.update(ast_argname(a) for a in al)

    def run(self):
        for item in self.f.body:
            self.visit(item)

    def is_iface_id(self,n):
        if isinstance(n,ast.Name) and n.id==self.iface_id:
            return True
        return isinstance(n,ast.Attribute) and self.is_iface_id(n.value)

    def visit_iface_id(self,x,attrs=None):
        if isinstance(x,ast.Name):
            nm=x.id
            if attrs!=None:
                nm+=attrs
            raise GrammaGrammarException('Forbidden access (%s) on line %d!!' % (nm, x.lineno))

        elif isinstance(x,ast.Attribute):
            p=ast_attr_path(x)
            attr=p[0].attr
            if attr=='random':
                self.uses_random=True
            elif attr=='sample':
                self.calls_sample=True
            elif attr=='state':
                self.statevars.add(p[1].attr)
            else:
                raise GrammaGrammarException('''forbidden use of %s.%s in gfunc %s of class %s on line %d''' % (self.iface_id,attr, self.f.name, self.target_class, x.lineno))

        else:
            raise GrammaGrammarException('iface_id not Attribute or Name? %s' % (x))

    def visit_AugAssign(self, ass):
        self.visit(ass.value)
        if self.is_iface_id(ass.target):
            self.visit_iface_id(ass.target)

    def visit_Assign(self, ass):
        self.visit(ass.value)
        for a in ass.targets:
            if self.is_iface_id(a):
                self.visit_iface_id(a)
            else:
                if isinstance(a,ast.Subscript):
                    if self.is_iface_id(a.value):
                        self.visit_iface_id(a.value)
                        return
                self.allowed_ids.add(a.id)

    def visit_Attribute(self,a):
        if self.is_iface_id(a):
            self.visit_iface_id(a)
        else:
            self.generic_visit(a)

    def visit_Name(self,n):
        if n.id==self.iface_id:
            self.visit_iface_id(n)
        elif not n.id in self.allowed_ids:
            raise GrammaGrammarException('''forbidden use of variable '%s' in %s on line %d
            of %s! to explicitly allow append to GrammaGrammar class variable
            list, ALLOWED_IDS. To prevent analysis of this function, add
            @gfunc(noauto=True) ''' % (n.id, self.f.name, n.lineno, self.target_class))

        # done
    def visit_FunctionDef(self,f):
        # don't descend!
        pass

class GFuncAnalyzeVisitor2(ast.NodeVisitor):
    '''
        A python AST node visitor for the reset_state method of a GrammaGrammar
        child class.

        This is used by analyze_gfuncs to checkthat all state spaces are
        initialized in reset_state.


        XXX: There isn't a way for a user to promise that some state space has
        been initialized without assinging it in the reset_state function.  More decorators?

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
            raise GrammaGrammarException('%s: initialize state fields in reset_state method of %s!' % (', '.join(throwme), self.target_class))

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
    classdef=ast.parse(s).body[0]
    def isgfuncdec(y):
        if isinstance(y,ast.Name) and y.id=='gfunc':
            return True
        return isinstance(y,ast.Call) and y.func.id=='gfunc'

    gfuncs=[x for x in classdef.body if isinstance(x,ast.FunctionDef) and any(isgfuncdec(y) for y in x.decorator_list)]
    statespace_usedby={}
    for gast in gfuncs:
        g=getattr(GrammaChildClass,gast.name)
        if g.noauto:
            continue

        #astpretty.pprint(gast)

        analyzer=GFuncAnalyzeVisitor(GrammaChildClass,gast,allowed_ids)
        analyzer.run()

        g.statevars.update(analyzer.statevars)
        g.uses_random=analyzer.uses_random
        g.calls_sample=analyzer.calls_sample

        for ss in g.statevars:
            statespace_usedby.setdefault(ss,set()).add(gast.name)

        #print(g)

    #print(statespace_usedby)
    reset_state_ast=([x for x in classdef.body if isinstance(x,ast.FunctionDef) and x.name=='reset_state']+[None])[0]
    if len(statespace_usedby.keys())>0 and reset_state_ast==None:
        # XXX enumerate which gfunc uses which statespace
        raise GrammaGrammarException('%s has no reset_state method, but uses statespaces %s' % (GrammaChildClass, statespace_usedby.keys()))

    if reset_state_ast != None:
        analyzer=GFuncAnalyzeVisitor2(GrammaChildClass,statespace_usedby,reset_state_ast)
        analyzer.run()

    

class GrammaGrammarType(type):
    '''
        metaclass that analyzes gfuncs of GrammaGrammar classes
    '''
    #def __new__(metaname, classname, baseclasses, attrs):
    #    return type.__new__(metaname, classname, baseclasses, attrs)

    def __init__(classobject, classname, baseclasses, attrs):
        analyze_gfuncs(classobject, getattr(classobject, 'ALLOWED_IDS', []))
        #print('done with analysis of %s' % classname)

class GrammaGrammar(with_metaclass(GrammaGrammarType,object)):
    '''
        The class defining functions and state management for a gramma and
        extensions.

        e.g. 
         g=GrammaGrammar('start:="a"|"b";')
         sampler=DefaultSampler(g)

         while True:
             sampler.reset()
             s=sampler.sample()
             print(s)
    '''

    # a child class variable that opens access to ids from gfuncs
    ALLOWED_IDS=[]

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
        state.d=0

    @gfunc
    def save_rand(x,n):
        x.random.save_state(n.as_str())
        yield ''

    @gfunc
    def load_rand(x,n):
        x.random.load_state(n.as_str())
        yield ''

    @gfunc
    def reseed_rand(x):
        x.random.seed(None)
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
        x.state.d+=1
        n=n.as_int()
        res=yield (c if x.state.d<=n else o)
        x.state.d-=1
        yield res

    def parse(self,gramma_expr_str):
        '''
            parse gramma_expr_sr with the current rules and functions and
            return a GExpr object
        '''
        lt=self.parser.parse('_:=(%s);' % gramma_expr_str)
        return GExpr.parse_larktree(lt.children[0].children[1])

    def generate(self,SamplerClass=DefaultSampler,samplerargs=(),startexpr=None):
        '''
            quick and dirty generator
        '''
        ctx=SamplerContext(self)
        sampler=SamplerClass(self,*samplerargs)
        if startexpr==None:
            startexpr=self.ruledefs['start']
        while True:
            yield ctx.sample(sampler,startexpr) 


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


class TracingSampler(object):
    '''
        we could wrap the stack object, but we'd need to know what we're
        pushing onto.
    '''
    __slots__='tracetree','base','random'

    def __init__(self,base,random):
        self.base=base
        self.random=random

    def reset(self):
        self.base.reset()
        self.tracetree=None

    def unwrap(self,top):
        return self.base.unwrap(top)

    def expr2ctor(self,ge):
        if self.tracetree==None:
            self.tracetree=TTNode(ge)
        else:
            self.tracetree=self.tracetree.add_child(ge)
        self.tracetree.inrand=self.random.r.get_state()
        return self.base.expr2ctor(ge)

    def complete(self,top,s):
        self.tracetree.s=s
        self.tracetree.outrand=self.random.r.get_state()
        if self.tracetree.parent!=None:
            self.tracetree=self.tracetree.parent

# vim: ts=4 sw=4


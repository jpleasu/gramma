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

    literals - same syntax as Python strings
        'this is a string'
        """this is a (possibly 
            multiline) string"""

    ternary operator (?:) - choice based on computed boolean
        `depth<5` ? x : y
        - the code term, `depth<5`, is a python expression testing the state
          variables 'depth'.  If the computed result is True, the result is
          x, else y.

    weighted alternation (|) - weighted random choice from alternatives
        2 x | 3.2 y | z
        - selects on of x, y, or z with probability 2/5.3, 2.3/5.3, and 1/5.3
          respectively.  The weight on z is implicitly 1.
        - omitting all weights corresponds to flat random choice, e.g.
            x | y | z
          selects one of x,y, or z with equal likelihood.
        - weights can also be code in backticks. For example:
            recurs := `depth<5` recurs | "token";
            - this sets the recurse branch weight to 1 (int(True)) if depth <5,
              and 0 otherwise.

    concatenation (.) - definite concatenation
        x . y

    repetition ({}) - random repeats
        x{3}
            - generate x exactly 3 times
        x{1,3}
            - generate a number n uniformly in [1,3] then generate x n times
        x{geom(3)}
            - sample a number n from a geometric distribution with mean 3, then
              generate x n times
        x{3,,geom(3)}
            - same as above, but reject n less than 3
        x{1,5,geom(3)}
            - same as above, but reject n outside of the interval [1,5]

     function call (gfuncs) - as defined in a GrammaGrammar subclass
        f(x)

        - by inheriting the GrammaGrammar class and adding decorated functions,
          GLF syntax can be extended.  See below.
        - functions can be stateful, meaning they rely on information stored in
          the SamplerInterface object.
        - evaluation is left to right.
        - functions aren't allowed to "look up" the execution stack, only back.

    TraceTree
    =========
    Sampling in Gramma is a form of expression tree evaluation where each node
    can use a random number generator.  E.g. to sample from

        "a" | "b"{1,5};

    The head of this expression, alternation, randomly chooses between its
    children, generating either "a" or a random sample of "b"{1,5} with equal
    odds.

    If "a" is chosen, the sample is complete.  If "b"{1,5} is chosen, its head,
    the repetition node, draws a random count between 1 and 5, then samples its
    child, "b", that number of times.  The results are concatenated, returned
    to the parent alternation node, and the sample is complete.

    Each possibility results in a different tree. Pictorially,

        alt       or         alt
         |                    |
        "a"                  rep
                            /...\
                           |     |
                          "b"   "b"

    The Tracer sideeffect computes this, so called, "trace tree", storing
    random number generator and other state when entering and exiting each
    node.

    Note: When sampling from a rule, the trace tree resembles the "recursion
    tree" of the recursion tree method for evaluating recursive programs. For
    example, a sample from

        r := "a" | "b" . r;
    
    could produce the trace tree

        alt
         |
        cat
        / \
       |   |
      "b" rule(r)
           |
          alt
           |
          "a"

    where the first alternation selected "b" . r, a concatenation whose
    righthand child samples the rule r recursively.
    

    Resampling
    ==========
    To produce strings similar to previous samples, we can hold fixed (or
    definitize) part of the corresponding trace tree.  By choosing to replay
    the context of execution for some but not all nodes of the trace tree, we
    can effectively create a template from a sample.

    The TraceNode object computed by the Tracer sideffect provides an interface
    for performing the operations presented below.

    - "resampling" starts with a tracetree and a node. For example, given the
      GExpr

            a(b(),c(),d())

        To record the random context around "c" we compute:

            save_rand('r0').a(b(),c().save_rand('r1'),d())

        and store r0 and r1.  The random number generator on entering "c" is
        then reseeded on entry, and resumed with r1 after exiting "c".

            load_rand('r0').a(b(), reseed_rand().c().load_rand('r1'), d())

        we could also choose the reseed explicitly via a load, e.g. if
        we'd saved a random state "r2" we could use:

            load_rand('r0').a(b(), load_rand(r2).c().load_rand('r1'), d())

    - to handle recursion, we must "unroll" rules until the point where the
      resampled node occurs.  e.g. the following generates arrows, "---->"
      with length (minus 1) distributed geometrically.

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

    Rope Aplenty
    ============
    GFuncs are (nearly) arbitraty code, so the analysis done by Gramma is
    necessarily limited.  To understand when Gramma might "give up", examples
    are given here.

    "Well behaved" GFuncs are generally subsumed by other constructs in Gramma,
    like ternary operators or dynamic alternations.  Heuristics, therefore,
    which would apply to "well behaved" gfuncs are avoided, assuming that
    grammars using gfuncs really need them.

    TraceNode.child_containing
    --------------------------
    .. treats gfuncs as atomic. a gfunc can modify the strings that it samples,
    so Gramma can't tell what parts of the string sampled from a gfunc are from
    what child.

    TraceNode.resample
    ------------------
    When a child node of a gfunc is resampled, Gramma tries to match previously
    sampled arguments with gexpr children of the original call, so that a new
    call can be constructed with appropriately sampled/definitized arguments.

        e.g. suppose we have

            choose(X,Y,Z)

        where

            @gfunc
            def choose(x, a, b, c):
                if (yield a)=='y':
                    yield (yield b)
                else:
                    yield (yield b)

        The tracetree records the sampled X and _depending on it's value_
        either the sampled Y or the sampled Z, but not both.

        If we resample (a child of) X, and suppose the original sample had
        chosen Y. Gramma will use the definitized sample of Y in the 2nd
        argument and the original expression for Z in the 3rd.

        If we resample (a child of) Y, Y must have been sampled.. so Z was
        not.. we will use the previous sample for X, which will again choose
        Y.. what we use in the 3rd argument doesn't matter in this case, but
        Gramma will use the original Z.

        If we resample X and Y, then it's possible that Z is sampled, since the
        1st arg might choose differently.

    If an argument is sampled more than once by a gfunc, that's a different story.
        
        suppose we have

            bigger("a"{0,5})

        where

            @gfunc
            def bigger(x, a):
                a1=(yield a)
                a2=(yield a)
                yield (a1 if len(a1)>len(a2) else a2)

        Suppose we resample the longer "a"{0,5} sample. Without replaying the
        previous sample, there's no way reproduce the function's behavior.  In
        this case, we therefore resample the entire argument.


    TODO:
        - analyze grammar to identify recursions.. insert parameters to
          alternations to help control depth.
        - examples
            - resample: sample, pick an "interesting" node, resample..
                - identify failing nodes by parsing compile error line
                  references.
                - AFL
                - hypothesis testing
                - function fitting
                - novelty search

            - regression: parameterize alternations w/ independent variables,
              compute dependent variable.

        - finish TraceNode.resample
            - when a gfunc is resampled, we should ensure it gets the same
              state as it would have in the original execution.

            - low level api allows decisions at each node:
                - on enter, use saved random, current random
                - on enter, use saved state or current state for each statevar
                - "definitize" - shorthand for use saved random and use saved
                  state
                - behavior might depend on child's behavior.. e.g. in the
                  simple resample case, a gfunc with a reampled child must be
                  executed again, meaning it might need to load the original
                  state first.

            - if we assume a new incoming random, we don't need to reseed random..
              and if the only users of random that follow are also being resampled,
              we don't need to reload previous random.
            - if we resample from a statevar using node, we will need to
              provide the state.. do we assume that successive resample nodes
              use the corresponding, possibly new, state, or are they all
              resumed w/ the state they were run with previously?
                - first off, drop any resample nodes that are children of other
                  resample nodes.. they won't exist in the new world!
                - it seems sane that distinct stateful nodes should resume with
                  the state they had last time.

        - stacked state
            - finish scoping exampling
            - to create scoped state, so that a statevar exists only at the
              depth it's scoped for, e.g.

                r := scoped(definevar(id) . "blah". (r . "blah" . usevar()){2,10} );

             where the GrammaGramma child class contains:

                def reset_state(state):
                    state.stk=[]
                    state.stktop=None

                @gfunc
                def scoped(x,child):
                    stk=x.state.stk
                    if len(stk)>0:
                        # inherit parent scopes values
                        stk.append(set(stk[-1]))
                    else:
                        stk.append(set())
                    x.state.stktop=stk[-1]
                    res = yield(child)
                    stk.pop()
                    x.state.stktop=stk[-1]
                    yield res

                @gfunc
                def definevar(x,id):
                    id=yield id
                    x.state.stktop.add(id)
                    yield id

                @gfunc
                def usevar(x):
                    yield random.choice(x.state.stktop)

            - we could also use a sideeffect to manage the stack, then just
              depend on the stack sideeffect and use it in gfuncs.
                def push(self,x,ge):
                    if is_our_node_type(ge):
                        x.state.stk.push..
                        return True
                    return False

                def pop(self,x,w,s):
                    if w:
                        x.state.stk.pop...

        - get_reset_states could attempt to guess the type of a state variable
          by finding the right hand side of assignments to it in reset_state.
          E.g.  set/dict/list.  That way, method access can be interpreted
          correctly.

        - analysis in GrammaGrammar constructor.
            - analyze GCode in parser, so that all calls to "parse" benefit
              from analysis
                - store analysis results on GExpr objects (and in grammar?)
            - the def, load, load_rand, and reseed_rand gfuncs are special.
                - load/def - the first string argument is the name of the state to
                  be def'd.


'''
from __future__ import absolute_import, division, print_function
import traceback
import sys
import time
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

    def perf_counter():
        return time.clock()

else:
    import builtins as __builtin__
    from functools import reduce

    def func_name(f):
        return f.__name__

    xrange=range

    def ast_argname(a):
        return a.arg

    def perf_counter():
        return time.perf_counter()


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

import logging

try:
    import astpretty
except ImportError:
    pass


import pysa

logging.basicConfig(format='%(asctime)-15s.%(msecs)d [%(name)s]: %(message)s', datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG)
log=logging.getLogger('gramma')

class GExprMetadata(object):
    '''
    state and rand metadata for use by TraceNode in resampling
    '''
    __slots__='statevar_defs','statevar_uses','uses_random','samples'

    def __init__(self,statevar_defs=None,statevar_uses=None,samples=False,uses_random=False):
        self.statevar_defs=set() if statevar_defs==None else statevar_defs
        self.statevar_uses=set() if statevar_uses==None else statevar_uses
        self.samples=samples
        self.uses_random=uses_random

    def __str__(self):
        l=[]
        if len(self.statevar_defs)>0:
            l.append('defs=%s' % ','.join(sorted(self.statevar_defs)))
        if len(self.statevar_uses)>0:
            l.append('uses=%s' % ','.join(sorted(self.statevar_uses)))
        if self.uses_random:
            l.append('uses_random')
        return ' '.join(l)
    def copy(self):
        return GExprMetadata(copy.deepcopy(self.statevar_defs), copy.deepcopy(self.statevar_uses), self.samples, self.uses_random)

    def __or__(self, other):
        return GExprMetadata(self.statevar_defs|other.statevar_defs,
                self.statevar_uses|other.statevar_uses,
                self.samples|other.samples, self.uses_random|other.uses_random)

GExprMetadata.DEFAULT=GExprMetadata()
GExprMetadata.DEFAULT_RANDOM=GExprMetadata(uses_random=True)

class GFuncWrap(object):
    __slots__='f','fname','analyzer','meta'
    def __init__(self,f,fname=None, analyzer=None, meta=None):
        self.f=f
        self.fname=fname
        self.analyzer=analyzer
        self.meta=GExprMetadata.DEFAULT.copy() if meta==None else meta

    def __call__(self,*l,**kw):
        return self.f(*l,**kw)

    def __str__(self):
        return 'gfunc %s%s %s' % (self.fname, ' %s' % self.analyzer if self.analyzer!=None else '', self.meta)

    def copy(self):
        return GFuncWrap(self.f, self.fname, self.analyzer, self.meta.copy())

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
    '''
    def _decorate(f,**kw):
        fname=kw.pop('fname',func_name(f))
        analyzer=kw.pop('analyzer',None)
        return GFuncWrap(f,fname,analyzer,GExprMetadata(**kw))

    if len(args)==0 or not callable(args[0]):
        return lambda f:_decorate(f,*args,**kw)

    f=args[0]
    return _decorate(f,**kw)


class GrammaParseError(Exception):
    pass

gramma_grammar = r"""
    ?start : ruledefs

    ruledefs : (ruledef|COMMENT)+

    ruledef : NAME ":=" tern ";"

    ?tern :  code "?" alt ":" alt | alt 

    ?alt : weight? cat ("|" weight? cat)*

    ?cat : rep ("." rep)*

    ?rep: atom ( "{" rep_args "}" )?

    rep_args : INT? (COMMA INT?)? (COMMA (func|code))?
            | func | code

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

    range : "[" CHAR  ".." CHAR "]"

    NAME : /[a-z_][a-z_0-9]*/

    string : STRING|CHAR|LONG_STRING

    code : /`[^`]*`/

    STRING : /[ubf]?r?("(?!"").*?(?<!\\)(\\\\)*?"|'(?!'').*?(?<!\\)(\\\\)*?')/i
    CHAR.2 : /'([^\\']|\\([\nrt']|x[0-9a-fA-F][0-9a-fA-F]))'/
    LONG_STRING.2: /[ubf]?r?("(?:"").*?(?<!\\)(\\\\)*?"(?:"")|'''.*?(?<!\\)(\\\\)*?''')/is

    COMMENT : /#[^\n]*/
    COMMA: ","

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

    def get_code(self):
        return []

    def get_meta(self):
        return GExprMetadata.DEFAULT

    def get_ancestor(self, cls):
        p=self.parent
        while p!=None:
            if isinstance(p,cls):
                return p
            p=p.parent
        return p

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

    def __str__(self):
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

    def __str__(self):
        return '%s(%s)' %(self.__class__.tag, ','.join(str(clt) for clt in self.children))

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

class GCode(GExpr):
    '''
       code expression, e.g. for dynamic alternation weights, ternary expressions, and dynamic repetition
    '''
    __slots__='expr','meta'
    def __init__(self, expr, meta=None):
        self.expr=expr
        self.meta=GExprMetadata.DEFAULT.copy() if meta==None else meta

    def invoke(self, x):
        locs=dict(x.params.__dict__, **x.state.__dict__)
        return eval(self.expr, globals(), locs)

    def __str__(self):
        return '`%s`' % self.expr

    def __add__(self,other):
        if isinstance(other,numbers.Number):
            return GCode('(%s)+%f' % (self.expr, other))
        return GCode('(%s)+(%s)' % (self.expr, other.expr))

    def copy(self):
        return GCode(self.expr, self.meta.copy())

class GTern(GInternal):
    tag='tern'

    __slots__='code',
    def __init__(self, code, children):
        GInternal.__init__(self,children)
        self.code=code

    def get_code(self):
        return [self.code]

    def get_meta(self):
        return self.code.meta

    def compute_case(self,x):
        return self.code.invoke(x)
 
    def __str__(self):
        return '%s ? %s : %s' % (self.code, self.children[0], self.children[1])

    def simplify(self):
        return GTern(self.code, [c.simplify() for c in self.children])

    @classmethod
    def parse_larktree(cls,lt):
        code=GCode(lt.children[0].children[0][1:-1])
        return cls(code,[GExpr.parse_larktree(clt) for clt in lt.children[1:]])

    def copy(self):
        return GTern(self.code, [c.copy() for c in self.children])

class defaultdict(dict):
    __slots__='default_func',
    def __init__(self,default_func):
        self.default_func=default_func

    def __missing__(self, key):
        return self.default_func(key)

class GAlt(GInternal):
    tag='alt'

    __slots__='weights','dynamic','nweights'
    def __init__(self, weights, children):
        GInternal.__init__(self,children)
        self.dynamic=any(w for w in weights if isinstance(w,GCode))

        if self.dynamic:
            self.weights=weights
        else:
            self.weights=weights
            w=np.array(weights)
            self.nweights=w/w.sum()

    def get_code(self):
        return [w for w in self.weights if isinstance(w,GCode)]

    def get_meta(self):
        return reduce(lambda a,b:a|b, (w.meta for w in self.weights if isinstance(w,GCode)), GExprMetadata(uses_random=True))

    def compute_weights(self,x):
        '''
            dynamic weights are computed using the SamplerInterface variable
            every time an alternation is invoked.
        '''
        if self.dynamic:
            w=np.array([w.invoke(x) if isinstance(w,GCode) else w for w in self.weights])
            return w/w.sum()
        return self.nweights
 
    def __str__(self):
        weights=[]
        for w in self.weights:
            if isinstance(w,GCode):
                weights.append('`%s` ' % w.expr)
            elif w==1:
                weights.append('')
            else:
                weights.append(str(w)+' ')
        s='|'.join('%s%s' % (w,c) for w,c in zip(weights, self.children))

        if self.parent!=None: #and isinstance(self.parent, (GCat, GRep)):
            return '(%s)' % s
        return s

    def simplify(self):
        if self.dynamic:
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

    def __str__(self):
        s='.'.join(str(cge) for cge in self.children)
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

    def get_code(self):
        if isinstance(self.dist, GCode):
            return [self.dist]
        return []

    def get_meta(self):
        if isinstance(self.dist, GCode):
            return self.dist.meta
        return GExprMetadata.DEFAULT_RANDOM

    @property
    def child(self):
        return self.children[0]

    def copy(self):
        return GRep([self.child.copy()],self.lo,self.hi,self.rgen,self.dist)

    def simplify(self):
        return GRep([self.child.simplify()],self.lo,self.hi,self.rgen,self.dist)

    def intargs(self):
        if self.lo==self.hi:
            return '%d' % (self.lo)
        lo='' if self.lo==0 else '%s' % (self.lo)
        return '%s,%d' % (lo,self.hi)

    def __str__(self):
        child=self.child
        if self.dist=='unif':
            return '%s{%s}' % (child, self.intargs())
        elif isinstance(self.dist, GCode):
            return '%s{%s,%s}' % (child, self.intargs(),self.dist)
        return '%s{%s,%s}' % (child, self.intargs(),self.dist)

    @classmethod
    def parse_larktree(cls,lt):
        child=GExpr.parse_larktree(lt.children[0])
        args=[GExpr.parse_larktree(c) for c in lt.children[1].children[:]]
        # figure out the distribution.. if not a GCode or a GFunc, assume uniform
        a=args[-1]
        if isinstance(a,GFunc):
            dist=a
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
            del args[-2:]
        elif isinstance(a, GCode):
            dist=a
            f=lambda lo,hi:lambda x:min(hi,max(lo,a.invoke(x)))
            del args[-2:]
        else:
            dist='unif'
            f=lambda lo,hi:lambda x:x.random.randint(lo,hi+1)

        #print('lenargs=%d' % len(args))

        # parse bounds
        if len(args)==0:
            # {`dynamic`}
            lo=0
            hi=2**32
        elif len(args)==1:
            # {2} or {2,`dynamic`}.. where the latter is pretty stupid
            lo=hi=args[0].as_int()
        elif len(args)==2:
            # {0,,`dynamic`}
            if(str(args[1])==','):
                lo=args[0].as_int()
                hi=2**32
            else:
                # {,2} or {,2,`dynamic`}
                if str(args[0])!=',':
                    raise GrammaParseError('expected comma in repetition arg "%s"' % lt)
                lo=0
                hi=args[1].as_int()
        elif len(args)==3:
            # {2,3} or {2,3,`dynamic`}
            lo=args[0].as_int()
            hi=args[2].as_int()
        else:
            raise GrammaParseError('failed to parse repetition arg "%s"' % lt)

        rgen=f(lo,hi)
        # print(lt)
        #print('lo=%d hi=%d\n\n' % (lo,hi))
        return GRep([child],lo,hi,rgen,dist)

class GRange(GExpr):
    tag='range'

    __slots__='lo','hi'
    def __init__(self,lo,hi):
        GExpr.__init__(self)
        self.lo=lo
        self.hi=hi

    def get_meta(self):
        return GExprMetadata.DEFAULT_RANDOM

    def copy(self):
        return GRange(self.lo,self.hi)

    def simplify(self):
        if self.hi-self.lo==1:
            return GTok.from_str(chr(self.lo))
        return self.copy()

    def __str__(self):
        return "['%s' .. '%s']" % (chr(self.lo), chr(self.hi))

    @classmethod
    def parse_larktree(cls,lt):
        lo=ord(GExpr.parse_larktree(lt.children[0]).as_str())
        hi=ord(GExpr.parse_larktree(lt.children[1]).as_str())
        return GRange(lo,hi)

class GFunc(GInternal):
    tag='func'

    __slots__='fname', 'gf'
    def __init__(self, fname, fargs, gf=None):
        GInternal.__init__(self,fargs)
        self.fname=fname
        # set in finalize_gexpr
        self.gf=gf

    def get_meta(self):
        return self.gf.meta

    def copy(self):
        return GFunc(self.fname,[c.copy() for c in self.fargs], self.gf)

    def simplify(self):
        return GFunc(self.fname,[c.simplify() for c in self.fargs], self.gf)

    @property
    def fargs(self):
        return self.children

    def __str__(self):
        return '%s(%s)' % (self.fname, ','.join(str(a) for a in self.children or self.fargs))

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

    __slots__='rname','rhs'
    def __init__(self,rname,rhs=None):
        GExpr.__init__(self)
        self.rname=rname
        self.rhs=rhs

    def copy(self):
        return GRule(self.rname, self.rhs)

    def is_rule(self, rname=None):
        if rname==None:
            return True
        return self.rname==rname

    def __str__(self):
        return self.rname

    @classmethod
    def parse_larktree(cls,lt):
        return GRule(lt.children[0].value)

for cls in GTern, GAlt, GCat, GRep, GFunc,   GRange, GRule:
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

    def set_cached_state(self,slot,val):
        self._cache[slot]=val

    def load(self,slot):
        '''
            set this random number generator state to the cached value 'slot'
        '''
        st=self._cache.get(slot)
        self.r.set_state(st)

    def save(self,slot):
        '''
            store the current random number generator state to 'slot'
        '''
        self._cache[slot]=self.r.get_state()

    def choice(self,l,p=None):
        return self.r.choice(l,p=p)

    def randint(self,low,high):
        return self.r.randint(low,high)

    def rand(self):
        return self.r.rand()

    def geometric(self,p):
        return self.r.geometric(p)

    def f(self,*l,**kw):
        print(l,kw)


class SamplerInterface(namedtuple('SamplerInterface','random state params')):
    '''
        constructed by GrammaSampler and passed to generators for access to
        random and state.
    '''

    def __new__(cls,sampler):
        return super(SamplerInterface,cls).__new__(cls,sampler.random,sampler.state,sampler.params)

class GrammaSampler(object):
    '''

        grammars provide grules, gfuncs, and the reset_state for its gfuncs.

        samplers mediate the GExpr requests and responses.

        the context manages state and executes the stack machine to generate
        samples.

    '''
    __slots__='grammar', 'transformers', 'sideeffects', 'state', 'random', 'stack', 'params', 'x'
    def __init__(self,grammar=None, **params):
        self.grammar=grammar
        self.transformers=[]
        self.sideeffects=[]
        self.random=GrammaRandom()
        self.state=GrammaState()
        self.params=type('Params',(),{})

        self.add_sideeffects(*self.grammar.sideeffects)


    def add_sideeffects(self,*sideeffects):
        for sideeffect in sideeffects:
            if inspect.isclass(sideeffect):
                sideeffect=sideeffect()
            self.sideeffects.append(sideeffect)

    def add_transformers(self,*transformers):
        for transformer in transformers:
            if inspect.isclass(transformer):
                transformer=transformer()
            self.transformers.append(transformer)

    def update_statecache(self, **kw):
        '''keywords are of the form slot=value'''
        self.state._cache.update(kw)

    def get_statecache(self):
        return self.state._cache

    def update_cache(self, cachecfg):
        self.state._cache.update(cachecfg.statecache)
        self.random._cache.update(cachecfg.randcache)

    def update_params(self, **kw):
        for k,v in kw.items():
            setattr(self.params,k,v)

    def reset_state(self):
        self.grammar.reset_state(self.state)

        for sideeffect in self.sideeffects:
            sideeffect.reset_state(self.state)

        self.x=SamplerInterface(self)
        self.stack=[]

    def sample(self,ge=None):
        if ge==None:
            ge=self.grammar.ruledefs['start']

        if isinstance(ge,string_types):
            ge=self.grammar.parse(ge)

        self.reset_state()

        a=ge
        while True:
            #assert(isinstance(a,GExpr))

            for transformer in self.transformers:
                a=transformer.transform(self.x,a)

            sideeffect_top=[sideeffect.push(self.x,a) for sideeffect in self.sideeffects]
            compiled_top=self.grammar.compile(a)(self.x)
            # wrapped top
            wtop=(sideeffect_top,compiled_top) 

            self.stack.append(wtop)

            a=next(compiled_top)

            while isinstance(a,string_types):
                #pop
                for sideeffect,w in zip(self.sideeffects,wtop[0]):
                    sideeffect.pop(self.x,w,a)

                self.stack.pop()

                if len(self.stack)==0:
                    return a

                wtop=self.stack[-1]
                a=wtop[1].send(a)


def get_reset_states(cls,method_name='reset_state'):
    m=pysa.get_method(cls,method_name)
    if m==None:
        return set()
    state_id=pysa.ast_argname(m.args.args[1])
    return set([n[1:] for n in pysa.get_defs(m) if n[0]==state_id])


class SideEffect(object):
    '''
        Base class for sampler sideeffects
    '''
    __slots__=()

    def get_reset_states(self):
        return get_reset_states(self.__class__)

    def reset_state(self,state):
        '''
        called before the stack machine starts
        '''
        pass

    def push(self,x,ge):
        '''
        when an expression is compiled to a coroutine, the stack machine
        pushes.  the return value is pushed at the same position in the stack,
        made available to pop later.
        '''
        return None

    def pop(self,x,w,s):
        '''
        when a coroutine complete, returning a string, s, it is popped.
        w is the value returned by the corresponding push.
        '''
        pass


class Transformer(object):
    '''
        an operation to perform on gexprs before being compiled in the sampler.
    '''
    __slots__=()
    def transform(self, x, ge):
        '''
        x is the SamplerInterface, ge is the incoming GExpr

        must return a gexpr.

        note: cache results!
        '''
        return ge

class GrammaGrammarException(Exception):
    pass

def ast_attr_path(x):
    p=[x]
    while isinstance(p[0].value,ast.Attribute):
        p.insert(0,p[0].value)
    return p


class GFuncAnalyzer(pysa.VariableAccesses):
    allowed_globals=['struct','True','False','None'] + [x for x in dir(__builtin__) if x.islower()]
    def __init__(self,target_class,gf,reset_states,param_ids=None,allowed_global_ids=None):
        pysa.VariableAccesses.__init__(self)
        self.target_class=target_class
        self.reset_states=reset_states

        self.allowed_ids=set(GFuncAnalyzer.allowed_globals)
        if allowed_global_ids!=None:
            self.allowed_ids.update(allowed_global_ids)
        self.param_ids=[] if param_ids==None else param_ids
        self.uses_random=False
        self.samples=False
        self.has_terminal_yield=False

        self.statevar_defs=set()
        self.statevar_uses=set()

        s=inspect.getsource(gf.f)
        s=textwrap.dedent(s)
        gf_ast=ast.parse(s).body[0]

        self.fname=gf_ast.name

        if len(gf_ast.args.defaults)>0:
            self._raise('''gfuncs mustn't use default argument values''')
        if gf_ast.args.kwarg!=None:
            self._raise('''gfuncs mustn't take keyword arguments''')

        fargs=[pysa.NamePath(a) for a in gf_ast.args.args]
        self.iface_id=fargs[0]
        self.extra_args=set(a.s for a in fargs[1:])

        self.allowed_ids.update(a.s for a in fargs)

        self.run(gf_ast)

        # .. and annotate GFuncWrap objects with state and rand metadata
        gf.meta.statevar_defs.update(self.statevar_defs)
        gf.meta.statevar_uses.update(self.statevar_uses)
        gf.meta.uses_random=self.uses_random
        gf.meta.samples=self.samples

        if not self.has_terminal_yield:
            self._raise('''doesn't yield a value''')

        #astpretty.pprint(gf_ast)
        #print(gf.meta)

    def is_iface_id(self,n):
        'n is a NamePath'
        return n[0].s==self.iface_id

    def _raise(self,msg):
        if hasattr(self,'stack') and len(self.stack)>0:
            raise GrammaGrammarException('''in line %d of gfunc %s of class %s: %s''' % (self.stack[-1].lineno, self.fname, self.target_class.__name__, msg))
        raise GrammaGrammarException('''gfunc %s of class %s: %s''' % (self.fname, self.target_class.__name__, msg))

    def defs(self, n, v):
        if self.is_iface_id(n):
            if n[1].s=='state':
                self.statevar_defs.add(n[2].s)
            elif n[1].s=='random' or n[1].s=='params':
                self._raise('forbidden access to SamplerInterface %s' % n[1:])
            else:
                self._raise('unexpected SamplerInterface field "%s", only "random", "state", and "params" are accessible' % n[1:].s)
        else:
            self.allowed_ids.add(n.s)

    def uses(self, n):
        if n.s in self.allowed_ids:
            return
        if self.is_iface_id(n):
            if n[1].s=='state':
                nn=n[2:]
                for s in self.reset_states:
                    if nn.begins(s):
                        self.statevar_uses.add(n[2].s)
                        break
                else:
                    self._raise('%s used without being initialized in any reset_state or explicitly allowed in allowed_global_ids or param_ids' % n.s)
            elif n[1].s=='params':
                if n[2]=='[]':
                    self._raise('params is not indexed, define ids with the param_ids argument of GrammaGrammar and set with update_params method of GrammaSampler')
                if not n[2] in self.param_ids:
                    self._raise('param "%s" not declared by grammar' % n[2].s)
            elif n[1].s=='random':
                self._raise('direct use of random object?')
            else:
                self._raise('unexpected SamplerInterface field "%s", only "random", "state", and "params" are accessible' % n[1:].s)
        else:
            self._raise('forbidden access to variable "%s"' % n.s)

    def mods(self, n, v):
        self.uses(n)
        self.defs(n,v)

    def calls(self, n, v):
        if self.is_iface_id(n):
            if n[1].s=='state':
                self.mods(n,v)
            elif n[1].s=='random':
                self.uses_random=True
            else:
                self._raise('forbidden all to variable "%s"' % n.s)

    def lambdas(self, l):
        pass

    def visit_Yield(self, y):
        self.visit(y.value)

        if any(n for n in self.stack[:-1] if isinstance(n,(ast.GeneratorExp,ast.ListComp))):
            self._raise('yield in a generator expression or list comprehension')

        p=self.stack[-2]
        if isinstance(p,ast.Call):
            if not y in p.args:
                self._raise('failed to analyze yield expression')
        elif isinstance(p,ast.BinOp):
            if p.left!=y and p.right!=y:
                self._raise('failed to analyze yield expression')
        else:
            if p.value!=y:
                self._raise('failed to analyze yield expression')

        if isinstance(p,ast.Expr):
            self.has_terminal_yield=True
        else:
            self.samples=True
            if not isinstance(y.value, ast.Name) or not y.value.id in self.extra_args:
                self._raise('gfuncs can only sample from their arguments')


class GCodeAnalyzer(pysa.VariableAccesses):
    def __init__(self, grammar, code):
        self.grammar=grammar
        self.allowed_ids=set(GFuncAnalyzer.allowed_globals) | set(self.grammar.allowed_global_ids) | set(self.grammar.param_ids)
        code.meta=GExprMetadata.DEFAULT.copy()
        self.code=code
        code_ast=ast.parse(code.expr).body[0]

        self.run(code_ast)

    def _raise(self,msg):
        raise GrammaGrammarException('''in gcode %s parsed by class %s: %s''' % (self.code, self.grammar.__class__.__name__, msg))

    def defs(self, n, v):
        self._raise('gcode cannot modify state')
    mods=defs

    def uses(self, n):
        if n.s in self.allowed_ids:
            return

        for s in self.grammar.reset_states:
            if n.begins(s):
                self.code.meta.statevar_uses.add(n[0].s)
                break
        else:
            self._raise('%s used without being initialized in any reset_state or explicitly allowed in allowed_global_ids or param_ids' % n.s)


def analyzer_use_first_arg(ge):
    ge.gf.meta.statevar_uses.add(ge.fargs[0].as_str())

def analyzer_def_first_arg(ge):
    ge.gf.meta.statevar_defs.add(ge.fargs[0].as_str())

class GrammaGrammar(object):
    '''
        The class defining functions and state management for a gramma and
        extensions.

        e.g. 
         g=GrammaGrammar('start:="a"|"b";')
         sampler=GrammaSampler(g)

         while True:
             print(sampler.sample())
    '''

    ruledef_parser = lark.Lark(gramma_grammar, parser='earley', lexer='standard', start='start')
    expr_parser = lark.Lark(gramma_grammar, parser='earley', lexer='standard', start='tern')

    __slots__='sideeffects', 'ruledefs', 'funcdefs', 'reset_states', 'allowed_global_ids', 'param_ids'

    def __init__(self, gramma_expr_str, sideeffects=None, param_ids=None, allowed_global_ids=None):
        '''
            gramma_expr_str - defines the grammar, including a start rule.

            sideeffects - a list of SideEffect objects or classes which the
            grammar, GFunc implementations or GCode expressions, require.

            param_ids - a list of param names that will be ignored by the GFunc
            and GCode analyzers.

            allowed_global_ids - a list of global variable identifiers ignored
            by GFunc and GCode analyzers.
        '''
        if sideeffects==None:
            sideeffects=[]

        # instantiate sideeffect classes
        self.sideeffects=[sideeffect() if inspect.isclass(sideeffect) else sideeffect for sideeffect in sideeffects]

        # analyze sideeffect state variable access
        reset_states=set()
        for se in self.sideeffects:
            reset_states|=se.get_reset_states()

        # analyze reset_state
        reset_states|=self.get_reset_states()

        cls=self.__class__
        allowed_global_ids=[] if allowed_global_ids==None else allowed_global_ids
        param_ids=[] if param_ids==None else param_ids

        self.funcdefs={}
        for n,gf in inspect.getmembers(self,predicate=lambda x:isinstance(x,GFuncWrap)):
            # make a grammar-local copy of gf
            gf=gf.copy()

            if gf.analyzer==None:
                GFuncAnalyzer(cls,gf,reset_states,param_ids=param_ids, allowed_global_ids=allowed_global_ids)

            self.funcdefs[gf.fname]=gf

        # record metadata
        self.reset_states=reset_states
        self.allowed_global_ids=allowed_global_ids
        self.param_ids=param_ids

        self.ruledefs={}
        self.add_rules(gramma_expr_str)

    def add_rules(self, ruledef_str):
        lt=GrammaGrammar.ruledef_parser.parse(ruledef_str)
        for ruledef in lt.children:
            self.ruledefs[ruledef.children[0].value]=GExpr.parse_larktree(ruledef.children[1])
        for ge in self.ruledefs.values():
            self.finalize_gexpr(ge)


    def get_reset_states(self):
        return get_reset_states(self.__class__)

    def reset_state(self,state):
        pass

    @gfunc
    def save_rand(x,slot):
        x.random.save(slot.as_str())
        yield ''

    @gfunc(uses_random=True)
    def load_rand(x,slot):
        x.random.load(slot.as_str())
        yield ''

    @gfunc(uses_random=True)
    def reseed_rand(x):
        x.random.seed(None)
        yield ''


    @gfunc(analyzer=analyzer_use_first_arg)
    def save(x,n,slot):
        x.state.save(n.as_str(),slot.as_str())
        yield ''

    @gfunc(analyzer=analyzer_def_first_arg)
    def load(x,n,slot):
        x.state.load(n.as_str(),slot.as_str())
        yield ''

    @gfunc(fname='def',analyzer=analyzer_def_first_arg)
    def def_(x,n,v):
        if isinstance(v,GTok):
            v=v.as_native()
        elif isinstance(v,GCode):
            v=v.invoke(x)
        setattr(x.state,n.as_str(), v)
        yield ''


    def finalize_gexpr(self, ge):
        '''
            grammar dependent finalization of a gexpr:
                1) compute meta for GCode nodes
                2) lookup GFuncs
                3) lookup rules

        '''
        if isinstance(ge,GFunc):
            if ge.gf!=None:
                # already finalized
                return
            gf=self.funcdefs.get(ge.fname,None)
            if gf==None:
                raise GrammaGrammarException('no gfunc named %s available to %s' % (ge.fname,self.__class__.__name__))
            ge.gf=gf
            if gf.analyzer!=None:
                ge.gf=ge.gf.copy()
                gf.analyzer(ge)

        elif isinstance(ge,GRule):
            if ge.rhs!=None:
                # already finalized
                return
            rhs=self.ruledefs.get(ge.rname,None)
            if rhs==None:
                raise GrammaGrammarException('no rule named %s available to %s' % (ge.rname,self.__class__.__name__))
            ge.rhs=rhs

        for code in ge.get_code():
            GCodeAnalyzer(self, code)

        if isinstance(ge, GInternal):
            for c in ge.children:
                self.finalize_gexpr(c)
        return ge

    def parse(self, gramma_expr_str):
        ''' gramma expression -> GExpr'''
        ge=GExpr.parse_larktree(GrammaGrammar.expr_parser.parse(gramma_expr_str))
        self.finalize_gexpr(ge)
        return ge

    def compile(self, ge):
        if isinstance(ge,GTok):
            def g(x):
                yield ge.as_str()
        elif isinstance(ge,GTern):
            def g(x):
                s=yield (ge.children[0] if ge.compute_case(x) else ge.children[1])
                yield s
        elif isinstance(ge,GAlt):
            if ge.dynamic:
                def g(x):
                    s=yield x.random.choice(ge.children,p=ge.compute_weights(x))
                    yield s
            else:
                def g(x):
                    s=yield x.random.choice(ge.children,p=ge.nweights)
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
            rhs=ge.rhs
            def g(x):
                s=yield(rhs)
                yield s
        elif isinstance(ge, GFunc):
            def g(x,gf=ge.gf,fargs=ge.fargs):
                return gf(x,*fargs)
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

class TraceNode(object):
    '''
        a node of the tracetree

        Tracer populates each TraceNode w/ incoming and outgoing rand and state
        values.
        - __when__
            - __it's_saved_to__

        - if a node samples
            - n.children
        - if a node uses_random,
            - on enter
                - n.inrand
            - before sampling child
                - child.inrand
            - after sampling child
                - child.outrand
            - on return
                - n.outrand
        - if a node uses state:
            - on enter
                - n.instate
            - after sampling child
                - child.outstate
        - if a node defs state:
            - before sampling child
                - child.instate
            - on return
                - n.outstate

        When resampling, we can "replay" different combinations of the above
        inputs to a node.

        - rand and other state can be set beforehand
            - random state can be set 
                load_rand('r').load('var','var0').e
        - (sub)samples
            - for all but gfuncs,

    '''
    __slots__='ge','parent','children','s','inrand','outrand','instate','outstate'
    def __init__(self,ge):
        self.ge=ge
        self.parent=None
        self.children=[]
        self.s=None
        self.inrand=None
        self.outrand=None
        self.instate=None
        self.outstate=None

    def add_child(self,ge):
        c=TraceNode(ge)
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

        if isinstance(self.ge, (GRule,GTern,GAlt)):
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


    def first(self, pred):
        for n in self.gennodes():
            if pred(n):
                return n
        return None

    def first_rule(self,rname):
        return self.first(lambda n:n.ge.is_rule(rname))

    def last(self, pred):
        for n in self.gennodesr():
            if pred(n):
                return n
        return None

    def gennodes(self):
        yield self
        for c in self.children:
            for cc in c.gennodes():
                yield cc

    def gennodesr(self):
        'reverse node generator'
        for c in reversed(self.children):
            for cc in c.gennodesr():
                yield cc
        yield self

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

    def resample_mostly(self,grammar,pred, factor=10):
        '''
            like resample, but non-resampled nodes aren't definitized, they're
            just biased toward their previous decision.

            factor is how much more likely the previous selection should be at
            each alternation and ternary.

            Even with a factor of 0, this resample is useful for preserving the
            tracetree structure for future resample operations.
        '''
        cachecfg=CacheConfig()

        def recurse(t):
            ''' 
                return resampled, gexpr
                    where resampled is True if a subexpression will be resampled
            '''
            if pred(t):
                outrand=t.last(lambda n:n.ge.get_meta().uses_random)
                if outrand!=None:
                    return True, t.ge.copy()

            ## mostly definitize ##
            ge=t.ge
            if isinstance(ge,(GAlt,GTern)):
                tc=t.children[0]
                b,tcc=recurse(tc)
                cmap=defaultdict(lambda c:(1,c))
                cmap[tc.ge]=(factor,tcc)

                weights,children=zip(*[cmap[c] for c in ge.children])
                return b,GAlt(weights,children)
            elif isinstance(ge,GRule):
                return recurse(t.children[0])
            elif isinstance(ge,(GCat,GRep)):
                l=[recurse(c) for c in t.children]
                return any(r for (r,ge) in l), GCat([ge for (r,ge) in l])
            elif isinstance(ge,GRange):
                return False, GTok.from_str(t.s)
            elif isinstance(ge,GTok):
                return False, ge.copy()
            elif isinstance(ge,GFunc):
                l=[recurse(c) for c in t.children]
                if not any(r for (r,ge) in l):
                    # preserve the function call
                    return False, GFunc(ge.fname,[arg[1] for arg in l],ge.gf)
                fargmap={}
                for i,c in enumerate(t.children):
                    fargmap.setdefault(c.ge,[]).append(i)
                args=[]
                for a in t.ge.fargs:
                    ta=fargmap.get(a,[])
                    if len(ta)>1:
                        log.warning('argument sampled multiple times in %s(..,%s,..): %s, resampling original expression' % (ge.fname,a,ta))
                        #log.warning(str(fargmap))
                        args.append(a.copy())
                    elif len(ta)==0:
                        # this argument wasn't sampled.. use a copy of the
                        # original
                        args.append(a.copy())
                    else:
                        # use the computed recursion on the tracenode child
                        args.append(l[ta[0]][1])
                return True, GFunc(ge.fname,args,ge.gf)
            else:
                raise ValueError('unknown GExpr node type: %s' % type(ge))

        b,ge=recurse(self)
        return ge.simplify(), cachecfg


    def resample(self,grammar,pred):
        '''
            computes ge, a GExpr that resamples only the nodes satisfying pred.

            computes cfg, a CacheConfig populated with any extra random and
            state values needed by ge.

            return ge,cfg
        '''

        cachecfg=CacheConfig()

        def recurse(t):
            ''' 
                return resampled, gexpr
                    where resampled is True if a subexpression will be resampled
            '''
            if pred(t):
                outrand=t.last(lambda n:n.ge.get_meta().uses_random)
                if outrand!=None:
                    ## resample ##
                    #slot=cachecfg.new_randstate(outrand.outrand)
                    #return GCat([grammar.parse('reseed_rand()'),t.ge.copy(),grammar.parse("load_rand('"+slot+"')")])
                    # assume "fresh" incoming random and no need to resume outgoing random
                    return True, t.ge.copy()
                    #return True, GCat([GTok.from_str('>>>>>>>'),t.ge.copy(),GTok.from_str('<<<<<<<<')])

            ## definitize ##
            ge=t.ge
            if isinstance(ge,GTern):
                return recurse(t.children[0])
            elif isinstance(ge,GAlt):
                return recurse(t.children[0])
            elif isinstance(ge,GRule):
                return recurse(t.children[0])
            elif isinstance(ge,(GCat,GRep)):
                l=[recurse(c) for c in t.children]
                return any(r for (r,ge) in l), GCat([ge for (r,ge) in l])
            elif isinstance(ge,GRange):
                return False, GTok.from_str(t.s)
            elif isinstance(ge,GTok):
                return False, ge.copy()
            elif isinstance(ge,GFunc):
                l=[recurse(c) for c in t.children]
                if not any(r for (r,ge) in l):
                    return False, GTok.from_str(t.s)
                fargmap={}
                for i,c in enumerate(t.children):
                    fargmap.setdefault(c.ge,[]).append(i)
                args=[]
                for a in t.ge.fargs:
                    ta=fargmap.get(a,[])
                    if len(ta)>1:
                        log.warning('argument sampled multiple times in %s(..,%s,..): %s, resampling original expression' % (ge.fname,a,ta))
                        #log.warning(str(fargmap))
                        args.append(a.copy())
                    elif len(ta)==0:
                        # this argument wasn't sampled.. use a copy of the
                        # original
                        args.append(a.copy())
                    else:
                        # use the computed recursion on the tracenode child
                        args.append(l[ta[0]][1])
                return True, GFunc(ge.fname,args,ge.gf)
            else:
                raise ValueError('unknown GExpr node type: %s' % type(ge))

        b,ge=recurse(self)
        return ge.simplify(), cachecfg


class DepthTracker(SideEffect):
    __slots__='pred','varname','initial_value'

    def __init__(self,pred=None, varname='depth', initial_value=0):
        if pred==None:
            pred=lambda ge:True
        self.pred=pred
        self.varname=varname
        self.initial_value=initial_value

    def get_reset_states(self):
        return set([pysa.NamePath(self.varname)])

    def reset_state(self,state):
        setattr(state, self.varname, self.initial_value)

    def push(self,x,ge):
        if self.pred(ge):
            setattr(x.state, self.varname, getattr(x.state, self.varname)+1)
            return True
        return False

    def pop(self,x,w,s):
        if w:
            setattr(x.state, self.varname, getattr(x.state, self.varname)-1)

class Tracer(SideEffect):
    __slots__='tt','tracetree'

    def reset_state(self,state):
        self.tracetree=None

    def push(self,x,ge):
        if self.tracetree==None:
            self.tt=self.tracetree=TraceNode(ge)
        else:
            self.tt=self.tt.add_child(ge)
        m=ge.get_meta()
        if m.uses_random:
            self.tt.inrand=x.random.r.get_state()
        if m.statevar_uses:
            self.tt.instate=type('',(),{})()
            for varname in m.statevar_uses:
                setattr(self.tt.instate, varname, copy.deepcopy(getattr(x.state,varname)))
        return None

    def pop(self,x,w,s):
        m=self.tt.ge.get_meta()

        self.tt.s=s
        if m.uses_random:
            self.tt.outrand=x.random.r.get_state()
        if m.statevar_defs:
            self.tt.outstate=type('',(),{})()
            for varname in m.statevar_defs:
                setattr(self.tt.outstate, varname, copy.deepcopy(getattr(x.state,varname)))

        self.tt=self.tt.parent

# vim: ts=4 sw=4

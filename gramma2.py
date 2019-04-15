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




    TODO:
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

        - GAlt
            - string should elide "1" weights.. use unnormalized integers if
              the representation is smaller.
            - alt(alt(),alt())  simplification

        - "resampling"
            - When a gexpr node is executed, it consumes information from the
              context and produces results.

                - if a node samples
                    - IN_sample return from sampling
                    - ? OUT effects from sampling?
                - if a node uses_random
                    - IN_rand on enter
                    - OUT_rand before sampling
                    - IN_rand after sampling
                    - OUT_rand on return
                - if a node uses state:
                    - IN_state on enter
                    - IN_state after sampling
                - if a node defs state:
                    - OUT_state before sampling
                    - OUT_state on return

            - The Tracer records all of this information for every node during
              the exectuion of a gexpr. 

        - state metadata should be useful to TraceNode, in particular the
          resample compiler.
            - "if we resample a node that had modified state, do we need to
              load that state on exit from the resample of S?  e.g. is it used
              later?"
            - low level TraceNode api:
                - find previous node, in preorder traversal, that
                    - defs a state
                    - defs a state that the current node uses
                    - effects random
                - find next node
                    - uses a state
                    - uses a state that the current node defs
                    - effects random
                - compute chains of consequence, forward and backward from a
                  given node.
                    - if n is resampled and defs state S
                        - if the next user of S is definite, then any further
                          resampled users must load state to resume.
                    - suppose n<m in preorder traversal, n defs a state that is
                      used by m and there is no node between n and m that uses
                      or defs that state.
                      - if both m and n are resampled, then we don't need to 


        - stacked state
            - to create scoped state, so that a statespace exists only at the
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

        - resample compilation
            - add params.. readonly state that isn't checked for initialization
            - during tracetree construction, store instate for use nodes and
              outstate for def nodes.
            - given a tracetree and a subset of nodes to resample, we compile
              it to a gexpr by definitizing and using "load", "load_rand", and
              "reseed_rand" calls.

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

'''
from __future__ import absolute_import, division, print_function
import traceback
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


import pysa

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
        return ('defs=%s' % ','.join(sorted(self.statevar_defs)) if len(self.statevar_defs)>0 else '')\
            + (' uses=%s' % ','.join(sorted(self.statevar_uses)) if len(self.statevar_uses)>0 else '')

    def copy(self):
        return GExprMetadata(copy.deepcopy(self.statevar_defs), copy.deepcopy(self.statevar_uses), self.samples, self.uses_random)

    def __or__(self, other):
        return GExprMetadata(self.statevar_defs|other.statevar_defs,
                self.statevar_uses|other.statevar_uses,
                self.samples|other.samples, self.uses_random|other.uses_random)

GExprMetadata.DEFAULT=GExprMetadata()
GExprMetadata.DEFAULT_RANDOM=GExprMetadata(uses_random=True)

class GFuncWrap(object):
    __slots__='f','fname','noauto','meta'
    def __init__(self,f,fname=None, noauto=False, meta=None):
        self.f=f
        self.fname=fname
        self.noauto=noauto
        self.meta=GExprMetadata.DEFAULT.copy() if meta==None else meta

    def __call__(self,*l,**kw):
        return self.f(*l,**kw)

    def __str__(self):
        return 'gfunc %s%s %s' % (self.fname, ' noauto' if self.noauto else '', self.meta)

    def copy(self):
        return GFuncWrap(self.f, self.fname, self.noauto, self.meta.copy())

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
        
        The fields of the SamplerInterface state used by a gfunc represent its
        "state space".  By tracking non-overlapping state spaces, Gramma can
        optimize certain operations.

        gfunc decorator keyword arguments

            fname = string
                provides a name other than the method name to use in glf

            noauto = True/False
                set to True to disable autoanalysis
            statevar_defs = list/set
            statevar_uses = list/set
                manual override for automatically inferred state def/use
            uses_random = True/False
                manual override for use of context random
    '''
    def _decorate(f,**kw):
        fname=kw.pop('fname',func_name(f))
        noauto=kw.pop('noauto',False)
        return GFuncWrap(f,fname,noauto,GExprMetadata(**kw))

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

    rep_args : INT? (COMMA INT)? (COMMA (func|code))?
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

    range : "[" ESCAPED_CHAR  ".." ESCAPED_CHAR "]"

    NAME : /[a-z_][a-z_0-9]*/

    string : ESCAPED_CHAR|STRING|LONG_STRING

    code : /`[^`]*`/

    STRING : /[ubf]?r?("(?!"").*?(?<!\\)(\\\\)*?"|'(?!'').*?(?<!\\)(\\\\)*?')/i
    ESCAPED_CHAR.2 : /'([^\']|\\([\nrt']|x[0-9a-fA-F][0-9a-fA-F]))'/
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

class GCode(GExpr):
    '''
       code expression, e.g. for dynamic alternation weights, ternary expressions, and dynamic repetition
    '''
    __slots__='expr','meta'
    def __init__(self, expr, meta=None):
        self.expr=expr
        self.meta=GExprMetadata.DEFAULT.copy() if meta==None else meta

    def __call__(self, state):
        return eval(self.expr, globals(), {k: getattr(state, k) for k in dir(state)})

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

    def compute_case(self,state):
        return self.code(state)
 
    def __str__(self,children=None):
        return '%s ? %s : %s' % (self.code, children[0], children[1])

    def simplify(self):
        return GTern(self.code, [c.simplify() for c in self.children])

    @classmethod
    def parse_larktree(cls,lt):
        code=GCode(lt.children[0].children[0][1:-1])
        return cls(code,[GExpr.parse_larktree(clt) for clt in lt.children[1:]])

    def copy(self):
        return GTern(self.code, [c.copy() for c in self.children])

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

    def get_code(self):
        return [w for w in self.weights if isinstance(w,GCode)]

    def get_meta(self):
        return reduce(lambda a,b:a|b, (w.meta for w in self.weights if isinstance(w,GCode)), GExprMetadata(uses_random=True))

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

    def __str__(self,children=None):
        child=children[0] if children else self.child
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
            f=lambda lo,hi:lambda x:min(hi,max(lo,a(x.state)))
            del args[-2:]
        else:
            dist='unif'
            f=lambda lo,hi:lambda x:x.random.randint(lo,hi+1)

        # parse bounds
        if len(args)==0:
            lo=0
            hi=2**32
        elif len(args)==1:
            # {2}
            lo=hi=args[0].as_int()
        elif len(args)==2:
            # {,2}
            if str(args[0])!=',':
                raise GrammaParseError('expected comma in repetition arg "%s"' % lt)
            lo=0
            hi=args[1].as_int()
        elif len(args)==3:
            # {2,3}
            lo=args[0].as_int()
            hi=args[2].as_int()
        else:
            raise GrammaParseError('failed to parse repetition arg "%s"' % lt)

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

    def get_meta(self):
        return GExprMetadata.DEFAULT_RANDOM

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


class SamplerInterface(namedtuple('SamplerInterface','random state param')):
    '''
        constructed by GrammaSampler and passed to generators for access to
        random and state.
    '''

    def __new__(cls,sampler):
        return super(SamplerInterface,cls).__new__(cls,sampler.random,sampler.state,sampler.param)

class GrammaSampler(object):
    '''

        grammars provide grules, gfuncs, and the reset_state for its gfuncs.

        samplers mediate the GExpr requests and responses.

        the context manages state and executes the stack machine to generate
        samples.

    '''
    __slots__='grammar', 'transformers', 'sideeffects', 'state', 'random', 'stack', 'param', 'x'
    def __init__(self,grammar=None, **param):
        self.grammar=grammar
        self.transformers=[]
        self.sideeffects=[]
        self.random=GrammaRandom()
        self.state=GrammaState()
        self.param=dict(param)

        self.add_sideeffects(*self.grammar.sideeffect_dependencies)

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

    def update_cache(self, cachecfg):
        self.state._cache.update(cachecfg.statecache)
        self.random._cache.update(cachecfg.randcache)

    def update_parms(self, **kw):
        self.param.update(kw)

    def reset_state(self):
        self.grammar.reset_state(self.state)

        for sideeffect in self.sideeffects:
            sideeffect.reset_state(self.state)

        self.x=SamplerInterface(self)
        self.stack=[]

    def sample(self,ge=None):
        if ge==None:
            #ge=self.grammar.ruledefs['start']
            ge='start'

        if isinstance(ge,string_types):
            ge=self.grammar.parse(ge)

        self.reset_state()

        a=ge
        while True:
            #assert(isinstance(a,GExpr))

            for transformer in self.transformers:
                a=transformer.transform(self.x,a)

            #push
            sideeffect_top=tuple(sideeffect.push(self.x,a) for sideeffect in self.sideeffects)
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

        must return a gexpr

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
    def __init__(self,target_class,gf,reset_states,allowed_ids=None):
        pysa.VariableAccesses.__init__(self)
        self.target_class=target_class
        self.reset_states=reset_states

        self.allowed_ids=set(GFuncAnalyzer.allowed_globals)
        if allowed_ids!=None:
            self.allowed_ids.update(allowed_ids)
        self.uses_random=False
        self.samples=False
        self.has_terminal_yield=False

        self.statevar_defs=set()
        self.statevar_uses=set()

        s=inspect.getsource(gf.f)
        s=textwrap.dedent(s)
        gf_ast=ast.parse(s).body[0]

        self.fname=gf_ast.name
        fargs=[pysa.NamePath(a) for a in gf_ast.args.args]
        self.iface_id=fargs[0]

        # XXX forbid args with default values
        self.allowed_ids.update(a.s for a in fargs)

        self.run(gf_ast)

        # .. and annotate GFuncWrap objects with state and rand metadata
        gf.meta.statevar_defs.update(self.statevar_defs)
        gf.meta.statevar_uses.update(self.statevar_uses)
        gf.meta.uses_random=self.uses_random
        gf.meta.samples=self.samples

        if not self.has_terminal_yield:
            self._raise('''doesn't yield a value''')


    def is_iface_id(self,n):
        'n is a NamePath'
        return n[0].s==self.iface_id

    def _raise(self,msg):
        if len(self.stack)>0:
            raise GrammaGrammarException('''in line %d of gfunc %s of class %s: %s''' % (self.stack[-1].lineno, self.fname, self.target_class.__name__, msg))
        raise GrammaGrammarException('''gfunc %s of class %s: %s''' % (self.fname, self.target_class.__name__, msg))

    def defs(self, n, v):
        if self.is_iface_id(n):
            if n[1].s=='state':
                self.statevar_defs.add(n[2].s)
            elif n[1].s=='random':
                self._raise('forbidden access to SamplerInterface %s' % n[1:])
            else:
                self._raise('unexpected SamplerInterface field "%s", only "random", "state", and "param" are accessible' % n[1:].s)
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
                    self._raise('%s used without being initialized in any reset_state' % n.s)
            elif n[1].s=='random':
                self.uses_random=True
            else:
                self._raise('unexpected SamplerInterface field "%s", only "random", "state", and "param" are accessible' % n[1:].s)
        else:
            self._raise('forbidden access to variable "%s"' % n.s)

    def mods(self, n, v):
        self.uses(n)
        self.defs(n,v)

    def calls(self, n, v):
        if self.is_iface_id(n):
            if n[1].s=='state':
                self.mods(n,v)

    def lambdas(self, l):
        pass

    def visit_Yield(self, y):
        self.visit(y.value)

        if any(n for n in self.stack[:-1] if isinstance(n,(ast.GeneratorExp,ast.ListComp))):
            self._raise('yield in a generator expression or list comprehension')

        p=self.stack[-2]
        if p.value!=y:
            self._raise('failed to analyze yield expression')

        if isinstance(p,ast.Expr):
            self.has_terminal_yield=True
        else:
            self.samples=True

class GCodeAnalyzer(pysa.VariableAccesses):
    def __init__(self, grammar, code):
        self.grammar=grammar
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
        if n.s in self.grammar.allowed_ids:
            return

        for s in self.grammar.reset_states:
            if n.begins(s):
                self.code.meta.statevar_uses.add(n[0].s)
                break
        else:
            self._raise('%s used without being initialized in any reset_state' % n.s)


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

    ALLOWED_GLOBAL_IDS=[]

    ruledef_parser = lark.Lark(gramma_grammar, parser='earley', lexer='standard', start='start')
    expr_parser = lark.Lark(gramma_grammar, parser='earley', lexer='standard', start='tern')

    __slots__='sideeffect_dependencies', 'ruledefs', 'funcdefs', 'reset_states', 'allowed_ids'

    def __init__(self, gramma_expr_str, sideeffect_dependencies=None):
        '''
            gramma_expr_str defines the grammar, including a start rule.

            sideeffect_dependencies is a list of SideEffect objects or classes
            which the grammar, GFunc implementations or GCode expressions,
            require.
        '''
        if sideeffect_dependencies==None:
            sideeffect_dependencies=[]

        # instantiate sideeffect classes
        self.sideeffect_dependencies=[sideeffect() if inspect.isclass(sideeffect) else sideeffect for sideeffect in sideeffect_dependencies]

        # analyze sideeffect state variable access
        reset_states=set()
        for se in self.sideeffect_dependencies:
            reset_states|=se.get_reset_states()

        # analyze reset_state
        reset_states|=self.get_reset_states()

        cls=self.__class__
        allowed_ids=getattr(cls, 'ALLOWED_GLOBAL_IDS', [])

        self.funcdefs={}
        for n,gf in inspect.getmembers(self,predicate=lambda x:isinstance(x,GFuncWrap)):
            # make a grammar-local copy of gf
            gf=gf.copy()

            if not gf.noauto:
                GFuncAnalyzer(cls,gf,reset_states,allowed_ids)

            self.funcdefs[gf.fname]=gf

        # record metadata
        self.reset_states=reset_states
        self.allowed_ids=allowed_ids

        self.ruledefs={}
        lt=GrammaGrammar.ruledef_parser.parse(gramma_expr_str)
        for ruledef in lt.children:
            rname=ruledef.children[0].value
            rvalue=GExpr.parse_larktree(ruledef.children[1])

            self.compute_meta(rvalue)

            self.ruledefs[rname]=rvalue

    def compute_meta(self, ge):
        for code in ge.get_code():
            GCodeAnalyzer(self, code)

        if isinstance(ge, GInternal):
            for c in ge.children:
                self.compute_meta(c)

    def get_reset_states(self):
        return get_reset_states(self.__class__)

    def reset_state(self,state):
        pass

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



    def parse(self, gramma_expr_str):
        '''
            parse a gramma expression.

            return a GExpr object
        '''
        ge=GExpr.parse_larktree(GrammaGrammar.expr_parser.parse(gramma_expr_str))
        self.compute_meta(ge)
        return ge

    def compile(self, ge):
        if isinstance(ge,GTok):
            def g(x):
                yield ge.as_str()
        elif isinstance(ge,GTern):
            def g(x):
                s=yield (ge.children[0] if ge.compute_case(x.state) else ge.children[1])
                yield s
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

class TraceNode(object):
    '''
        a node of the tracetree
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
                lastrand=t.last(lambda n:n.ge.get_meta().uses_random)
                if lastrand!=None:
                    slot=cachecfg.new_randstate(lastrand.outrand)
                    return GCat([grammar.parse('reseed_rand()'),t.ge.copy(),grammar.parse("load_rand('"+slot+"')")])

            ge=t.ge
            if isinstance(ge,GTern):
                # XXX lost sample, rand stream out of sync w/ original
                return recurse(t.children[0])
            elif isinstance(ge,GAlt):
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

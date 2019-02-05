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
    - repetition ({}) - randome repeats
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
          the GrammaSampler object.
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

    The _defining_ constaint on ("a"|"b"){1,2} that produced "ab" is
        rep(n=2,child1=alt(c=1,child="a"), child2=alt(c=2,child="b"))
    By omitting the constraint on child1 of the root rep, we get
        rep(n=2,child2=alt(c=2,child="b"))
    In this case, we can compile this constrained expression to
        ("a"|"b")."b"




    TODO:
        - GExpr as data
            - (bi)simulation is done by the sampler
            - sampler can modify recursion in house.
            - recursive sample calls from gfunc can't be prescribed.. external
              entropy and/or state control?
              - if state spaces can be deep-copied, we can manufacture
                dependencies w/out re-execution.

        - fix the multiple random object situation in the sampler.. e.g. is
          there every a need to 'externally' manage entropy?

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
#from builtins import (bytes, str, open, super, range,zip, round, input, int, pow, object)

# builtins' object fucks up slots
from builtins import (bytes, str, open, super, range,zip, round, input, int, pow)

import sys
#sys.setrecursionlimit(20000)
sys.setrecursionlimit(200000)

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

class GFuncInterface(namedtuple('GFuncInterface','sample random state')):
    '''
        constructed by GrammaSampler and passed to GFunc as first argument.
    '''

    def __new__(cls,sampler):
        return super(GFuncInterface,cls).__new__(cls,sampler.sample,sampler.random,sampler.state)

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
                        x.sample - the GrammaSampler sample method
                        x.random - an alias for x.sampler.random

                    args are the arguments of f as GExpr elements.  In
                    particular, "constants" are are type GTok and must be
                    converted, and generate GEXpr objects can be sampled from.

            *) mustn't access global variables
            *) may store state as fields of the GrammaState instance, state
            *) mustn't take additional keyword arguments, only "grammar",
                "sampler", and "state" are allowed.
            *) may sample from GExpr arguments using the GrammaSampler instance, sampler
            *) may access entropy from the GrammaSampler instance, sampler
        
        The fields of the GrammaSampler object used by a gfunc reprsent its
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
        return GFuncWrap(f,fname=f.func_name,**kw)

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
            raise GParseError, '''unrecognized Lark node %s during parse of glf''' % lt
        return cls.parse_larktree(lt)
 

    def copy(self):
        return None

    def simplify(self):
        'copy self.. caller must ultimately set parent attribute'
        return self.copy()

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
            raise GrammaParseError, 'not a num: %s' % self

    @staticmethod
    def from_ltok(lt):
        return GTok(lt.type,lt.value)

    @staticmethod
    def join(tok_iter):
        return GTok('string',repr(''.join(t.as_str() for t in tok_iter)))

    def is_stateful(self,x,assume_no=None):
        return False

    @staticmethod
    def new_empty():
        return GTok('string',repr(''))

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

        for w,c in self.zip(self.weights, self.children):
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
                raise GrammaParseError, 'no dist %s' % (fname)

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
            return GTok('string', repr(chr(self.lo)))
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

class GrammaSamplerException(Exception):
    pass

class GrammaState(object):
    pass

class GrammaSampler(object):
    '''
        samplers execute GExprs, providing state and random.

        the grammar will provide gfuncs and reset_state function.
        
    '''
    __slots__='grammar','state','random'
    def __init__(self,grammar):
        self.grammar=grammar
        self.random=np.random.RandomState()
        self.state=GrammaState()

    def reset(self):
        self.grammar.reset_state(self.state)

    def sample(self,ge):
        if isinstance(ge,GTok):
            return ge.as_str()
        elif isinstance(ge, GAlt):
            return self.sample(self.random.choice(ge.children,p=ge.weights))
        elif isinstance(ge, GCat):
            return ''.join(self.sample(cge) for cge in ge.children)
        elif isinstance(ge, GRep):
            return ''.join(self.sample(ge.child) for _ in xrange(ge.rgen(self)))
        elif isinstance(ge, GRange):
            return chr(self.random.randint(ge.lo,ge.hi+1))
        elif isinstance(ge, GFunc):
            return self.grammar.funcdefs[ge.fname](GFuncInterface(self),*ge.fargs)
        elif isinstance(ge, GRule):
            return self.sample(self.grammar.ruledefs[ge.rname])

        raise GrammaSamplerException('unrecognized expression: %s' % ge)


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
    import __builtin__
    allowed_globals=['struct','True','False','None'] + [x for x in dir(__builtin__) if x.islower()]

    def __init__(self,target_class,f,allowed_ids=None):
        self.target_class=target_class
        # id of the GFuncInterface (first argument of gfunc)
        # other argument ids
        self.allowed_ids=set(GFuncAnalyzeVisitor.allowed_globals)
        if allowed_ids!=None:
            self.allowed_ids.update(allowed_ids)
        self.uses_random=False
        self.calls_sample=False
        self.statevars=set()

        self.f=f
        al=f.args.args
        self.iface_id=al[0].id
        # XXX prevent args with default values?
        self.allowed_ids.update(a.id for a in al)

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
    '''
    def __init__(self,target_class,statespace_usedby,f):
        self.target_class=target_class
        self.statespace_usedby=statespace_usedby
        self.f=f
        al=f.args.args
        self.state_id=al[1].id
        self.assigned_statespaces=set()

    def run(self):
        for item in self.f.body:
            self.visit(item)
        for uid,fs in self.statespace_usedby.iteritems():
            if not uid in self.assigned_statespaces:
                if len(fs)==1:
                    fss="gfunc '%s'" % next(iter(fs))
                else:
                    fss="gfuncs {%s}" % (','.join("'%s'" % fn for fn in fs))
                raise GrammaGrammarException('''statespace '%s' is used by %s in class %s but not set in reset_state!''' %(uid,fss,self.target_class))

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

class GrammaGrammar(object):
    '''
       e.g. 
        g=GrammaGrammar('start:="a"|"b";')
        x=GrammaSampler(g)

        for s in x.generate(): print(s)
    '''
    __metaclass__=GrammaGrammarType


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
        res=x.sample(c if x.state.d<=n else o)
        x.state.d-=1
        return res

    def parse(self,gramma_expr_str):
        '''
            parse gramma_expr_sr with the current rules and functions and
            return a GExpr object
        '''
        lt=self.parser.parse('_:=%s;' % gramma_expr_str)
        return GExpr.parse_larktree(lt.children[0].children[1])

    def generate(self,SamplerClass=GrammaSampler,samplerargs=(),startexpr=None):
        '''
            yield random_state, string
        '''
        sampler=SamplerClass(self,*samplerargs)
        if startexpr==None:
            startexpr=self.ruledefs['start']
        while True:
            rst=x.random.get_state()
            self.reset_state(sampler.state)
            yield rst, sampler.sample(startexpr) 


# vim: ts=4 sw=4


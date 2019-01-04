#!/usr/bin/env python
'''

    A Gramma language file defines the syntax for a stochastic expression
    system.  The Gramma class parses and evaluates expressions.  By adding
    methods in a subclass, new functions can be added.

    Gramma expressions represent random variables with string type.
    Expressions are built from literals, operators, and functions.
        - literals - same syntax as Pytnon strings
            'this is a string'
            """so is this"""
        - concatenation (.) - definite concatenation
            a . b
        - alternation (|) - random choice from alternatives
            a | b
        - functions - as defined in Gramma child class
            f(arg)

    Sampling a Gramma expression is done by _stochastic_ evaluation.  The
    RichSampler will store the intermediate definite results of the evaluation.
    A RichSample can be _relaxed_ at select nodes in order to produce a similar
    random variable.

    TODO:
        - ReplacingSampler and the "replace" function.
            - expressions composed of stochastic operations that don't
              influence state are definite strings.
            - stateful operations must be re-executed with the same incoming
              state in order to recreate the same (sub)result.
              - if functions used _labeled_ state, we could define a scope and
                avoid computation. If all nodes below the sup of nodes using a
                given label were definite, we could make those functions
                definite.

                f(x).x.X.x.g(x)

        - document weird effects
            - with functions:
                @gfunc
                def f(x):
                    x.value=x.random.random_sample() >.5:
                    return ''
                @gfunc
                def g(x,arg1,arg2):
                    if x.value:
                        return x.sample(arg1)
                    else:
                        return x.sample(arg2)
            - and grammar
                A := f();
                B := g('1','2');
                start := A . B;
            - if we choose to resample A while leaving B "definite", what do we
              expect to see?
                - we should get an x.value as set randomly by f
                - we will execute g because it's stateful, but where should its
                  sample come from?  we should have to re-sample arg1 or arg2,
                  whichever is chosen implicitly by f.. The replacing sampler
                  will be wrong half the time.

        


        abcde

                r1
          r2         r3
         [0,2)      [2,5)
         ab         cde

  support(e) = { k!00, k!10, k!20 }


'''

import sys
#sys.setrecursionlimit(20000)
sys.setrecursionlimit(200000)

import inspect

import lark

import numpy as np

from itertools import islice,groupby

from collections import namedtuple

from functools import wraps

_gfunc_defaults=dict(
        stateful=True
)
def gfunc(*args,**kw):
    '''
        a Gramma function decorator
    '''
    kw=dict(_gfunc_defaults, **kw)

    def _decorate(f,**kw):
        kw=dict(_gfunc_defaults,**kw)
        @wraps(f)
        def g(x,*l,**kw):
            return f(x,*l,**kw)
        g.is_gfunc=True
        g.fname=kw.get('fname',f.func_name)
        g.stateful=kw.get('stateful')
        return g

    if len(args)==0 or not callable(args[0]):
        if len(args)>=1:
            name=args[0]
        else:
            name=kw.get('name')
        return lambda f:_decorate(f,**kw)
    f=args[0]
    kw['fname']=f.func_name
    return _decorate(f,**kw)

def decorator(original_function=None, optional_argument1=None, optional_argument2=None):
    def _decorate(function):
        @wraps(function)
        def wrapped_function(*args, **kwargs):
            return wrapped_function
    if original_function:
        return _decorate(original_function)
    return _decorate


class GrammaParseError(Exception):
    pass

gramma_grammar = r"""
    ?start : ruledefs

    ruledefs : (ruledef|COMMENT)+

    ruledef : NAME ":=" alt ";"

    ?alt : cat ("|" cat)*

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

class GTree(object):
    '''
        the expression tree for a Gramma expression.
    '''
    __slots__=['parent']
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

    # tag2cls[lark_tree_node.data]=GTree_with_parse_larktree_method
    tag2cls={}

    @classmethod
    def parse_larktree(cls,lt):
        if isinstance(lt,lark.lexer.Token):
            return GTok(lt.type,lt.value)
        if lt.data=='string':
            return GTok('string',lt.children[0].value)

        cls=GTree.tag2cls.get(lt.data)
        if cls==None:
            raise GParseError, '''unrecognized Lark node %s during parse of Gramma''' % lt
        return cls.parse_larktree(lt)
 

    def copy(self):
        return None

    def simplify(self):
        'copy self.. caller must ultimately set parent attribute'
        return self.copy()

class GTok(GTree):
    __slots__=['type','value']
    def __init__(self,type,value):
        GTree.__init__(self)
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

    def do_sample(self,x):
        return self.as_str()

    def as_num(self):
        if self.type==u'INT':
            return int(self.value)
        elif self.type==u'FLOAT':
            return float(self.value)
        else:
            raise ValueError, 'not a num: %s' % self

    @staticmethod
    def join(tok_iter):
        return GTok('string',repr(''.join(t.as_str() for t in tok_iter)))

    def is_stateful(self,x,assume_no=None):
        return False

    @staticmethod
    def new_empty():
        return GTok('string',repr(''))

class GInternal(GTree):
    '''
        nodes with gtree children
    '''

    # internal nodes must have a tag, corresponding to the larktree data field

    __slots__=['children']
    def __init__(self, children):
        GTree.__init__(self)
        self.children=children
        for c in self.children:
            c.parent=self

    def __str__(self,children=None):
        return '%s(%s)' %(self.__class__.tag, ','.join(str(cpt) for cpt in children or self.children))

    @classmethod
    def parse_larktree(cls,lt):
        return cls([GTree.parse_larktree(cpt) for cpt in lt.children])
    
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

    def do_sample(self,x):
        return x.sample(x.random.choice(self.children))

    def __str__(self,children=None):
        s='|'.join(str(cgt) for cgt in children or self.children)
        if self.parent!=None and isinstance(self.parent, (GCat, GRep)):
            return '(%s)' % s
        return s

    def is_stateful(self,x,assume_no=None):
        return any(c.is_stateful(x,assume_no) for c in self.children)

    def simplify(self):
        children=self.flat_simple_children()

        # dedupe (and sort) by string representation
        children=[v for k,v in sorted(dict( (str(c),c) for c in children).items())]
        #haveit=set()
        #l=[]
        #for c in children:
        #    s=str(c)
        #    if not s in haveit:
        #        l.append(c)
        #        haveit.add(s)
        #children=l

        if len(children)==0:
            return GTok.new_empty()
        if len(children)==1:
            return self.children[0].simplify()
        return GAlt(children)

class GCat(GInternal):
    tag='cat'

    def do_sample(self,x):
        return ''.join(x.sample(cgt) for cgt in self.children)

    def __str__(self,children=None):
        s='.'.join(str(cgt) for cgt in children or self.children)
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

    __slots__=['rgen', 'lo', 'hi', 'dist']

    def __init__(self,children,lo,hi,rgen,dist):
        GInternal.__init__(self,children)
        self.lo=lo
        self.hi=hi
        self.rgen=rgen
        self.dist=dist

    @property
    def child(self):
        return self.children[0]

    def do_sample(self,x):
        return ''.join(x.sample(self.child) for _ in xrange(self.rgen(x)))

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
        child=GTree.parse_larktree(lt.children[0])
        args=[GTree.parse_larktree(c) for c in lt.children[1].children[:]]
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
                raise ValueError, 'no dist %s' % (fname)

            f=lambda lo,hi:lambda x:min(hi,max(lo,g(x)))
            args.pop()
        else:
            dist='unif'
            f=lambda lo,hi:lambda x:x.random.randint(lo,hi+1)
        lo=0 if len(args)==0 else args.pop(0).as_int()
        hi=2**32 if len(args)==0 else args.pop(0).as_int()

        rgen=f(lo,hi)
        return GRep([child],lo,hi,rgen,dist)

class GRange(GTree):
    tag='range'

    __slots__=['lo','hi']
    def __init__(self,lo,hi):
        GTree.__init__(self)
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

    def do_sample(self,x):
        return chr(x.random.randint(self.lo,self.hi+1))

    def __str__(self,children=None):
        return "['%s' .. '%s']" % (chr(self.lo), chr(self.hi))

    @classmethod
    def parse_larktree(cls,lt):
        lo=ord(GTree.parse_larktree(lt.children[0]).as_str())
        hi=ord(GTree.parse_larktree(lt.children[1]).as_str())
        return GRange(lo,hi)

class GFunc(GInternal):
    tag='func'

    __slots__=['fname']
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

    def do_sample(self,x):
        return x.funcdefs[self.fname](x,*self.fargs)
    
    def __str__(self,children=None):
        return '%s(%s)' % (self.fname, ','.join(str(a) for a in children or self.fargs))

    @classmethod
    def parse_larktree(cls,lt):
        fname=lt.children[0].value

        if len(lt.children)>1:
            fargs=[GTree.parse_larktree(c) for c in lt.children[1].children]
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

class GRule(GTree):
    'this is a _reference_ to a rule.. the rule definition is part of the Gramma class'

    tag='rule'

    __slots__=['rname']
    def __init__(self,rname):
        GTree.__init__(self)
        self.rname=rname

    def copy(self):
        return GRule(self.rname)

    def do_sample(self,x):
        return x.sample(x.ruledefs[self.rname])

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
    GTree.tag2cls[cls.tag]=cls



class RichSample(GTree):
    '''
        The complete evaluation record produced during GTree sampling,
        including entropy at each node.

        Convenience methods for correlating the resulting string sample with
        the grammatical origin are also available.

    '''
    def __init__(self,ogt,parent=None):
        self.ogt=ogt

        # the RichSample hierarchy constructed during the sample
        self.parent=parent
        self.children=[]

        self.s=None

        # the random state when ogt was sampled to produce s
        self.inrand=None

        # the random state on completion of the ogt sample
        self.outrand=None


        # the offset of s from the start of the root's s
        self.off=None

    def visit(self,f):
        f(self)
        for c in self.children:
            c.visit(f)

    def leaves(self):
        l=[]
        def f(n):
            if len(n.children)==0:
                l.append(n)
        self.visit(f)
        return l

    def genwalk(self):
        yield self
        for c in self.children:
            for n in c.genwalk():
                yield n

    def gen_rule_nodes(self, rname):
        for n in self.genwalk():
            if n.ogt.isrule(rname):
                yield n

    def compute_offsets(self,off=0):
        self.off=off
        for c in self.children:
            c.compute_offsets(off)
            off+=len(c.s)

    def __str__(self):
        # display inrand information about repeat or not.. in particular, expand rules that have inrands
        return self.ogt.__str__(self.children)

    def do_sample(self,x):
        if x.using_default_random:
            # using the original distribution with default random
            #print('default %s' % self.ogt)
            return self.ogt.do_sample(x) 
        else:
            if self.inrand==None:
                x.using_default_random=True
                #print('enable default_random with %s and sampler %s' % (self.ogt, x))
                res=self.do_sample(x)
                x.using_default_random=False
                x.random.set_state(self.outrand)
                return res
            else:
                #print('repeating %s, should get children %s' % (self.ogt, self.children))
                x.random.set_state(self.inrand)
                return self.ogt.do_sample(ReplacingSampler(x,self.children)) # when ogt calls x.sample, it will get self.children in order

    def needs_sampling(self,x,assume_no=None):
        '''
            RichSample nodes need to be resampled if either they are stateful
            or they use any random.. e.g. inrand==None
        '''
        if self.inrand==None:
            return True
        if isinstance(self.ogt,(GTok,GRange)):
            return False
        if isinstance(self.ogt,GFunc):
            if assume_no==None:
                assume_no=set()
            if self.ogt.fname in assume_no:
                return False
            #XXX function and rule collision
            assume_no.add(self.ogt.fname)
            return x.funcdefs[self.ogt.fname].stateful or any(c.needs_sampling(x,assume_no) for c in self.children)
        if isinstance(self.ogt,GRule):
            if assume_no==None:
                assume_no=set()
            if self.ogt.rname in assume_no:
                return False
            assume_no.add(self.ogt.rname)
            return any(c.needs_sampling(x,assume_no) for c in self.children)
        return any(c.needs_sampling(x,assume_no) for c in self.children)

    def to_gtree(self,x,assume_constant=None):
        '''
            create on ordinary gtree that can be resampled again..

            XXX: parent attributes haven't been set
        '''

        if self.inrand==None:
            return self.ogt.copy()

        if not self.needs_sampling(x,assume_constant):
            return GTok('string', repr(self.s))

        if isinstance(self.ogt,GCat):
            return GCat([c.to_gtree(x,assume_constant) for c in self.children])
        if isinstance(self.ogt,GAlt):
            return self.children[0].to_gtree(x,assume_constant)
        if isinstance(self.ogt,GRange):
            x.random.set_state(self.inrand)
            c=self.ogt.do_sample(x)
            return GTok('string',repr(c))
        if isinstance(self.ogt,GRule):
            return self.children[0].to_gtree(x,assume_constant)
        if isinstance(self.ogt,GRep):
            if len(self.children)==0:
                return GTok.new_empty()
            if len(self.children)==1:
                return self.children[0].to_gtree(x,assume_constant)
            return GCat([c.to_gtree(x,assume_constant) for c in self.children])

        if isinstance(self.ogt,GFunc):
            # we need any entropy drawn in the func to remain the same
            # maybe by wrapping it while unwrapping children??
            # we don't even know how the func will sample its arguments..
            # possibly multiple times..  The wrapper would look exactly like
            # the ReplacingSampler above. How do we add that the to Gramma syntax?
            #
            #   replace(f(a,b,c), randstate, alt_samples))
            #     maybe
            #   replace(f, a,b,c,  randstate,  alt_samples))
            #   elements should represent random variables.. so writing
            #   f(a,b,c) as the argument to replace is wrong.  We are really passing the function named 'f'
            if self.needs_sampling(x,assume_constant):
                return GFunc('replace', [self.ogt, GRule('randstate###')] + [c.to_gtree(x,assume_constant) for c in self.children])
            else:
                return GTok('string',repr(self.s))
        raise GParseError, '''can't convert %s''' % self.ogt

       


class Sampler(object):
    __slots__=['base']
    def __init__(self,base):
        object.__setattr__(self,'base',base)

    def __getattr__(self,a):
        return getattr(self.base,a)

    def __setattr__(self,a,v):
        setattr(self.base,a,v)

    def reset(self):
        self.base.reset()

    def sample(self,gt):
        return gt.do_sample(self)

class ReplacingSampler(Sampler):
    def __init__(self,base, children):
        Sampler.__init__(self,base)
        # instance attribute, do not set at base
        object.__setattr__(self,'chit',iter(children))

    def sample(self,gt):
        s=self.chit.next().do_sample(self.base)
        return s

class RichSampler(Sampler):
    def __init__(self,base):
        Sampler.__init__(self,base)

    def sample(self,gt):
        'build a RichSample while sampling'
        p=self.stack[-1]
        r=RichSample(gt,p)
        p.children.append(r)

        self.stack.append(r)
        r.inrand=self.random.get_state()
        r.s=gt.do_sample(self)
        r.outrand=self.random.get_state()
        self.stack.pop()

        return r.s

    def reset(self):
        Sampler.reset(self)

        self.root=RichSample('root')
        self.stack=[self.root]

    def build(self,randstate=None):
        'returns a RichSample'

        if randstate==None:
            self.random.set_state(randstate)
        self.reset()
        self.sample(self.ruledefs['start'])
        r=self.root.children[0]
        r.compute_offsets()
        return r


class Gramma:
    '''
        A gramma parsetree represents a distribution on strings.

        A gramma language definition file defines a set of trees, vis a vis, the rules.

        x=Gramma(language_definition)

        For any tree, gt, defined by a gramma language definition, x.sample(gt)
        samples a single string.


        The methods of Gramma (and subclasses) are invoked recursively by
        sample with parse tree children and expect a string result.
    '''
    parser = lark.Lark(gramma_grammar, parser='earley', lexer='standard')

    def __init__(self, gramma_text):
        lt=self.parser.parse(gramma_text)

        self.ruledefs={}
        for ruledef in lt.children:
            rname=ruledef.children[0].value
            rvalue=GTree.parse_larktree(ruledef.children[1])
            self.ruledefs[rname]=rvalue

        self.funcdefs={}
        for n,f in inspect.getmembers(self,predicate=inspect.ismethod):
            if hasattr(f,'is_gfunc'):
                self.funcdefs[f.fname]=f.__func__

        self.default_random=np.random.RandomState()
        self.temp_random=np.random.RandomState()
        self.using_default_random=False

    @property
    def random(self):
        #print('getting random, default=%s' % self.using_default_random)
        if self.using_default_random:
            return self.default_random
        else:
            return self.temp_random

    def reset(self):
        self.d=0

    def simple_sample(self,gt):
        return gt.do_sample(self)
    
    def generate(self,startrule=None):
        '''
            yield random_state, string
        '''
        x=Sampler(self)
        if startrule==None:
            startrule=x.ruledefs['start']
        while True:
            st=x.random.get_state()
            x.reset()
            yield st, x.sample(startrule) 

    def build_richsample(self,randstate=None):
        'build a RichSample object'
        return RichSampler(self).build(randstate)

    def rsample(self,gt):
        'do the resample - sample replacement'
        r=self.stack[-1].next()
        if r.inrand!=None:
            self.random.set_state(r.inrand)

            self.stack.append(iter(r.children))
            res=self.simple_sample(r.gt)
            self.stack.pop()
        else:
            self.random.seed()
            self.sample=self.simple_sample
            res=self.simple_sample(r.gt)
            self.sample=self.rsample
        return res

    def gen_resamples(self,r):
        'assume inrand has been cleared for resampler nodes that should be stochastic'
        startrule=self.ruledefs['start']
        self.sample=self.rsample
        while True:
            self.reset()
            self.stack=[iter([r])]
            yield self.sample(startrule)



    @gfunc
    def rlim(self,c,n,o):
        '''
            recursion limit - if recursively invoked to a depth < n times,
                sample c, else sample o

            e.g. 

                R :=  rlim("a" . R, 3, "");

                produces "aaa" 
            
        '''
        self.d+=1
        n=n.as_int()
        res=self.sample(c if self.d<=n else o)
        self.d-=1
        return res




class Example(Gramma):
    g1=r'''
        start := words . " " . ['1'..'9'] . digit{geom(5)};

        digit := ['0' .. '9'];

        words := ( "stink" | "stank" );
    '''

    g2=r'''
        start := x;
        x := "a" | "b" . x;
    '''


    def __init__(x):
        Gramma.__init__(x,Example.g1)

def test_example():
    g=Example()
    it=g.generate()
    for i in xrange(10):
        print(it.next()[1])

def test_parser():
    global t
    parser = Lark(gramma_grammar, parser='earley', lexer='standard')
    #    start := ( "a" | "b" ) . ['\xaa'..'\xbb'];
    g1=r'''
        start :=  digits{1}
                | digits{.2}
                | digits{3,.4}
                | digits{5,6}
                | digits{7,8,.9}
        ;

        digits := ['1' .. '9'];

    '''
    g2=r'''
        start := x{0,1,g(10)};

        x := x;
    '''
    t=parser.parse(g2)

    print(t)

if __name__ == '__main__':
    #test_parser()
    test_example()

# vim: ts=4 sw=4

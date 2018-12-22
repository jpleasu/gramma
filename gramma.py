#!/usr/bin/env python
'''

    A Gramma language file defines the syntax for a stochastic expression
    system.  The Gramma class parses and evaluates expressions.  By adding
    methods in a subclass, new functions can be added.

    TODO:


        abcde

                r1
          r2         r3
         [0,2)      [2,5)
         ab         cde

  support(e) = { k!00, k!10, k!20 }

    My real problem is that I want Gramma to be a sample with a recursive
    function that I can change .. selectively?



'''

import sys
#sys.setrecursionlimit(20000)
sys.setrecursionlimit(200000)

import inspect

import lark

import numpy as np

from itertools import islice

from collections import namedtuple

from functools import wraps

def gfunc(*args,**kw):
    '''
        a Gramma function decorator
    '''
    def _decorate(f,name):
        @wraps(f)
        def g(x,*l,**kw):
            return f(x,*l,**kw)
        g.is_gfunc=True
        g.fname=name
        return g

    if len(args)==0 or not callable(args[0]):
        if len(args)>=1:
            name=args[0]
        else:
            name=kw.get('name')
        return lambda f:_decorate(f,name=name)
    f=args[0]
    return _decorate(f,name=f.func_name)

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


class GTree(object):
    '''
        wrapper for Gramma parsetree nodes and derived generators
    '''
    __slots__=['parent']
    def __init__(gt,parent):
        gt.parent=parent

    def get_ancestor(gt, name):
        p=gt.parent
        while p!=None:
            if p.name==name:
                return p
            p=p.parent
        return p

class GTok(GTree):
    __slots__=['type','value']
    def __init__(gt,parent,type,value):
        GTree.__init__(gt,parent)
        gt.type=type
        gt.value=value

    def __str__(gt):
        return repr(gt.value)

    def as_int(gt):
        return int(gt.value)

    def as_str(gt):
        return eval(gt.value)

    def sample(gt,x):
        return gt.as_str()

    def as_num(gt):
        if gt.type==u'INT':
            return int(gt.value)
        elif gt.type==u'FLOAT':
            return float(gt.value)
        else:
            raise ValueError, 'not a num: %s' % gt


class GInternal(GTree):
    __slots__=['name','children']
    def __init__(gt,parent,name,ptchildren):
        GTree.__init__(gt,parent)
        gt.name=name
        gt.children=[parse_generator(cpt,gt) for cpt in ptchildren]

    def __str__(gt):
        return '%s(%s)' %(gt.name, ','.join(str(cpt) for cpt in gt.children))

class GAlt(GInternal):
    def __init__(gt,parent,ptchildren):
        GInternal.__init__(gt,parent,'alt',ptchildren)

    def sample(gt,x):
        return x.sample(x.random.choice(gt.children))

class GCat(GInternal):
    def __init__(gt,parent,ptchildren):
        GInternal.__init__(gt,parent,'cat',ptchildren)

    def sample(gt,x):
        return ''.join(x.sample(cgt) for cgt in gt.children)

class GRep(GInternal):
    __slots__=['rgen']

    def __init__(gt,parent,ptchildren):
        GInternal.__init__(gt,parent,'rep',ptchildren)
        args=gt.children[1].children[:]
        a=args[-1]
        if (not isinstance(a,GTok)) and a.name==u'func':
            fname=a.children[0].value
            if len(a.children)>1:
                ptargs=a.children[1].children
            else:
                ptargs=[]
            ptargs=[a.as_num() for a in ptargs]
            if fname==u'geom':
                # "a"{geom(n)} has an average of n copies of "a"
                parm=1/float(ptargs[0]+1)
                g=lambda x: x.random.geometric(parm)-1
            elif fname=='norm':
                g=lambda x:int(x.random.normal(*ptargs)+.5)
            elif fname=='binom':
                g=lambda x:x.random.binomial(*ptargs)
            elif fname=='choose':
                g=lambda x:x.random.choice(ptargs)
            else:
                raise ValueError, 'no dist %s' % (fname)

            f=lambda lo,hi:lambda x:min(hi,max(lo,g(x)))
            args.pop()
        else:
            f=lambda lo,hi:lambda x:x.random.randint(lo,hi+1)
        lo=0 if len(args)==0 else args.pop(0).as_int()
        hi=2**32 if len(args)==0 else args.pop(0).as_int()

        gt.rgen=f(lo,hi)

    def sample(gt,x):
        return ''.join(x.sample(gt.children[0]) for _ in xrange(gt.rgen(x)))


class GRange(GInternal):
    def __init__(gt,parent,ptchildren):
        GInternal.__init__(gt,parent,'range',ptchildren)
        gt.lo=ord(gt.children[0].as_str())
        gt.hi=ord(gt.children[1].as_str())

    def sample(gt,x):
        return x.random.choice([chr(v) for v in range(gt.lo,gt.hi+1)])

class GFunc(GInternal):
    __slots__=['fname','fargs']
    def __init__(gt,parent,ptchildren):
        GInternal.__init__(gt,parent,'func',ptchildren)
        gt.fname=gt.children[0].value

        if len(gt.children)>1:
            gt.fargs=gt.children[1].children
        else:
            gt.fargs=[]

    def sample(gt,x):
        return x.func_trees[gt.fname](x,*gt.fargs)

class GRule(GInternal):
    __slots__=['rname']
    def __init__(gt,parent,ptchildren):
        GInternal.__init__(gt,parent,'rule',ptchildren)
        gt.rname=gt.children[0].value

    def sample(gt,x):
        return x.sample(x.rule_trees[gt.rname])


def parse_generator(pt,parent=None):
    if isinstance(pt,lark.lexer.Token):
        return GTok(parent,pt.type,pt.value)
    if pt.data=='alt':
        return GAlt(parent,pt.children)
    if pt.data=='cat':
        return GCat(parent,pt.children)
    if pt.data=='rep':
        return GRep(parent,pt.children)
    if pt.data=='range':
        return GRange(parent,pt.children)
    if pt.data=='func':
        return GFunc(parent,pt.children)
    if pt.data=='rule':
        return GRule(parent,pt.children)
    if pt.data=='string':
        return GTok(parent,'string',pt.children[0].value)
    return GInternal(parent,pt.data,pt.children)


class Random:
    def __init__(self):
        pass

    def __getattr__(self,a):
        return getattr(np.random,a)

class RichSample:
    '''
        a GTree sample with everthing that went in to generating it.

        This structure is used to identify and resample subtrees.
    '''
    def __init__(self,gt,parent=None):
        self.gt=gt

        # the RichSample hierarchy constructed during the sample
        self.parent=parent
        self.children=[]

        self.s=None
        # the random state when gt was sampled to produce s
        self.inrand=None
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

    def compute_offsets(self,off=0):
        self.off=off
        for c in self.children:
            c.compute_offsets(off)
            off+=len(c.s)


class Sampler(object):
    def __init__(self,base):
        self.base=base

    def __getattr__(self,a):
        return getattr(self.base,a)

    def reset(x):
        x.base.reset()

    def sample(x,gt):
        return gt.sample(x)

class RichSampler(Sampler):
    def __init__(self,base):
        self.base=base

    def sample(x,gt):
        'build a RichSample while sampling'
        p=x.stack[-1]
        r=RichSample(gt,p)
        p.children.append(r)

        x.stack.append(r)
        r.inrand=x.random.get_state()
        r.s=gt.sample(x)
        x.stack.pop()

        return r.s

    def reset(x):
        Sampler.reset(x)

        x.root=RichSample('root')
        x.stack=[x.root]

    def build(x,randstate=None):
        'returns a RichSample'

        if randstate==None:
            x.random.set_state(randstate)
        x.reset()
        x.sample(x.rule_trees['start'])
        r=x.root.children[0]
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

    def __init__(x, gramma_text):
        pt=x.parser.parse(gramma_text)
        gt=parse_generator(pt)

        x.rule_trees={}
        for ruledef in gt.children:
            name=ruledef.children[0].value
            expr_tree=ruledef.children[1]
            x.rule_trees[name]=expr_tree

        x.func_trees={}
        for n,f in inspect.getmembers(x,predicate=inspect.ismethod):
            if hasattr(f,'is_gfunc'):
                x.func_trees[f.fname]=f.__func__

        x.random=Random()

    def reset(x):
        x.d=0

    def simple_sample(x,gt):
        'straight sample, no additional side effects'
        return gt.sample(x)
    
    def generate(x):
        '''
            yield random_state, string
        '''
        x=Sampler(x)
        startrule=x.rule_trees['start']
        while True:
            st=x.random.get_state()
            x.reset()
            yield st, x.sample(startrule) 

    def build_richsample(x,randstate=None):
        'build a RichSample object'
        return RichSampler(x).build(randstate)

    def rsample(x,gt):
        'do the resample - sample replacement'
        r=x.stack[-1].next()
        if r.inrand!=None:
            x.random.set_state(r.inrand)

            x.stack.append(iter(r.children))
            res=x.simple_sample(r.gt)
            x.stack.pop()
        else:
            x.random.seed()
            x.sample=x.simple_sample
            res=x.simple_sample(r.gt)
            x.sample=x.rsample
        return res

    def gen_resamples(x,r):
        'assume inrand has been cleared for resampler nodes that should be stochastic'
        startrule=x.rule_trees['start']
        x.sample=x.rsample
        while True:
            x.reset()
            x.stack=[iter([r])]
            yield x.sample(startrule)



    @gfunc
    def rlim(x,c,n,o):
        '''
            recursion limit - if recursively invoked to a depth < n times,
                sample c, else sample o

            e.g. 

                R :=  rlim("a" . R, 3, "");

                produces "aaa" 
            
        '''
        x.d+=1
        n=n.as_int()
        res=x.sample(c if x.d<=n else o)
        x.d-=1
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

#!/usr/bin/env python
'''

    A Gramma language file defines the syntax for a stochastic expression
    system.  The Gramma class parses and evaluates expressions.  By adding
    methods in a subclass, new functions can be added.

    TODO:
        - construct a sample tree that stores
            a) the tree structure
            b) the entropy sufficient to recreate the result


        - expose random methods from Gramma that record for later playback
        - sample subtrees while holding everthing else fixed.

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

from itertools import islice

from collections import namedtuple


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

def mkgen(f):
    '''
    decorator to create a generator which generates repeated calls
        e.g. f=mkgen(f) <=> f=iter
    '''
    def gf(x,*its):
        while True: 
            yield f(x,*[it.next() for it in its])
    return gf



class Node(object):
    __slots__=['parent']
    def __init__(self,parent):
        self.parent=parent

    def get_ancestor(self, name):
        p=self.parent
        while p!=None:
            if p.name==name:
                return p
            p=p.parent
        return p

class Tok(Node):
    __slots__=['value']
    def __init__(self,parent,value):
        Node.__init__(self,parent)
        self.value=value

    def __str__(self):
        return repr(self.value)


class Rule(Node):
    __slots__=['name','children']
    def __init__(self,parent,name,ptchildren):
        Node.__init__(self,parent)
        self.name=name
        self.children=[convert_parsetree(c,self) for c in ptchildren]

    def __str__(self):
        return '%s(%s)' %(self.name, ','.join(str(c) for c in self.children))

class Resampler:
    def __init__(self,et,parent=None):
        self.et=et
        self.parent=parent
        self.children=[]
        self.s=None
        self.inrand=None
        self.outrand=None
        self.offset=None

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

def convert_parsetree(pt,parent=None):
    if isinstance(pt,lark.lexer.Token):
        return Tok(parent,pt.value)
    return Rule(parent,pt.data,pt.children)


class Random:
    def __init__(self):
        pass

    def __getattr__(self,a):
        return getattr(np.random,a)


etfunc_t=namedtuple('etfunc_t','name args')
def etfunc(et):
    name=et.children[0].value
    if len(et.children)>1:
        args=et.children[1].children
    else:
        args=[]
    return etfunc_t(name,args)

etrule_t=namedtuple('etrule_t', 'name')
def etrule(et):
    name=et.children[0].value
    return etrule_t(name)

class Gramma:
    '''
        A gramma parsetree represents a distribution on strings.

        A gramma language definition file defines a set of trees, vis a vis, the rules.

        x=Gramma(language_definition)

        For any tree, et, defined by a gramma language definition, x.sample(et)
        samples a single string.


        The methods of Gramma (and subclasses) are invoked recursively by
        sample with parse tree children and expect a string result.
    '''

    parser = lark.Lark(gramma_grammar, parser='earley', lexer='standard')

    def __init__(x, gramma_text):
        pt=x.parser.parse(gramma_text)
        et=convert_parsetree(pt)

        x.rule_trees={}
        for ruledef in et.children:
            name=ruledef.children[0].value
            expr_tree=ruledef.children[1]
            x.rule_trees[name]=expr_tree

        x.stack=[]

        x.random=Random()

    def getstring(x,et):
        'convert string element to python string'
        if et.name==u'string':
            l=et.children[0]
        else:
            l=et
        return eval(l.value)

    def getint(x,et):
        return int(et.value)
    def getnum(x,et):
        'convert int or float element to python equivalent'
        if et.type==u'INT':
            return int(et.value)
        elif et.type==u'FLOAT':
            return float(et.value)
        else:
            raise ValueError, 'not a num: %s' % et

    def choose(x,*ets):
        return x.sample(x.random.choice(ets))

    def concat(x,*ets):
        return ''.join(x.sample(et) for et in ets)

    def rep(x,et,*args):
        args=list(args)
        a=args[-1]
        if (not isinstance(a,Tok)) and a.name==u'func':
            fname=a.children[0].value
            if len(a.children)>1:
                fargs=a.children[1].children
            else:
                fargs=[]
            fargs=[x.getnum(a) for a in fargs]
            if fname==u'geom':
                # "a"{geom(n)} gets an average of n copies of "a"
                # argument is the average number of 
                n=x.random.geometric(1/float(fargs[0]+1))-1
            elif fname=='norm':
                n=int(x.random.normal(*fargs)+.5)
            elif fname=='binom':
                n=x.random.binomial(*fargs)
            elif fname=='choose':
                n=x.random.choice(fargs)
            else:
                raise ValueError, 'no dist %s' % (fname)

            f=lambda lo,hi:min(hi,max(lo,n))
            args.pop()
        else:
            f=lambda lo,hi:x.random.randint(lo,hi+1)
        lo=0 if len(args)==0 else x.getint(args.pop(0))
        hi=2**32 if len(args)==0 else x.getint(args.pop(0))

        #print('lo=%d hi=%d' % (lo,hi))
        n=f(lo,hi)

        s=''.join(x.sample(et) for _ in xrange(n))
        return s

    def rlim(x,c,n,o):
        '''
            recursion limit - if recursively invoked to a depth < n times,
                sample c, else sample o

            e.g. 

                R :=  rlim("a" . R, 3, "");

                produces "aaa" 
            
        '''
        x.d+=1
        n=x.getint(n)
        res=x.sample(c if x.d<=n else o)
        x.d-=1
        return res

    def osample(x,et):
        '''
            fully stochastic sample function
        '''

        if et.name==u'alt': # choice
            return x.choose(*et.children)
        if et.name==u'cat':
            return x.concat(*et.children)
        if et.name==u'rep':
            return x.rep(et.children[0], *et.children[1].children)
        if et.name==u'range':
            lo=ord(eval(et.children[0].value))
            hi=ord(eval(et.children[1].value))
            return x.random.choice([chr(v) for v in range(lo,hi+1)])
        if et.name==u'string':
            return eval(et.children[0].value)
        if et.name==u'func':
            f=etfunc(et)
            return getattr(x,f.name)(*f.args)
        if et.name==u'rule':
            r=etrule(et)
            return x.sample(x.rule_trees[r.name])
        else:
            raise GrammaParseError, '''can't transform %s''' % et
    
    def generate(x):
        x.sample=x.osample
        startrule=x.rule_trees['start']
        while True:
            x.reset()
            yield x.sample(startrule) 

    def reset(x):
        x.d=0
        del x.stack[:]

    def br_sample(x,et):
        'resampler structure builder sample function'
        p=x.stack[-1]
        r=Resampler(et,p)
        p.children.append(r)

        x.stack.append(r)
        r.inrand=x.random.get_state()
        r.s=x.osample(et)
        x.stack.pop()

        return r.s

    def buildresampler(x):
        'build a resampler object'
        startrule=x.rule_trees['start']
        x.sample=x.br_sample
        x.reset()
        root=Resampler('root')
        x.stack.append(root)
        x.sample(startrule)
        r=root.children[0]
        r.compute_offsets()
        return r

    def rsample(x,et):
        'do the resample - sample replacement'
        r=x.stack[-1].next()
        if r.inrand!=None:
            x.random.set_state(r.inrand)

            x.stack.append(iter(r.children))
            res=x.osample(r.et)
            x.stack.pop()
        else:
            x.random.seed()
            x.sample=x.osample
            res=x.osample(r.et)
            x.sample=x.rsample
        return res

    def gen_resamples(x,r):
        'assume inrand has been cleared for resampler nodes that should be stochastic'
        startrule=x.rule_trees['start']
        x.sample=x.rsample
        while True:
            x.reset()
            x.stack.append(iter([r]))
            yield x.sample(startrule)



class Example(Gramma):
    g1=r'''
        start := words . " " . ['1'..'9'] . digit{0,5.};

        digit := ['0' .. '9'];

        words := ( "stink" | "stank" );
    '''

    g2=r'''
        start := x;
        x := "a" | x;
    '''


    def __init__(x):
        Gramma.__init__(x,Example.g2)

def test_example():
    g=Example()
    it=g.generate()
    for i in xrange(10):
        print(it.next())

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
    test_parser()
    #test_example()

# vim: ts=4 sw=4

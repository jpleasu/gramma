#!/usr/bin/env python
'''

    A Gramma language file defines the syntax for a stochastic expression
    system.  The Gramma class parses and evaluates expressions.  By adding
    methods in a subclass, new functions can be added.

    TODO:
        - store the evaluation tree as it's constructed -- make it available to
          functions -- before and after invoking children will correspond to
          pre and post visition positions in depth first walk of expression tree.
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
sys.setrecursionlimit(2000000)

import inspect

import random

from lark import Lark
from lark.lexer import Token

import numpy as np


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

    parser = Lark(gramma_grammar, parser='earley', lexer='standard')

    def __init__(x, gramma_text):
        x.t=x.parser.parse(gramma_text)

        x.rule_trees={}
        for ruledef in x.t.children:
            name=ruledef.children[0].value
            expr_tree=ruledef.children[1]
            x.rule_trees[name]=expr_tree

        x.stack=[]

    def getstring(x,et):
        'convert string element to python string'
        if et.data==u'string':
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

    def lit(x,s):
        return s

    def choose(x,*ets):
        return x.sample(random.choice(ets))

    def concat(x,*ets):
        return ''.join(x.sample(et) for et in ets)

    def rep(x,et,*args):
        args=list(args)
        a=args[-1]
        if (not isinstance(a,Token)) and a.data==u'func':
            fname=a.children[0].value
            if len(a.children)>1:
                fargs=a.children[1].children
            else:
                fargs=[]
            fargs=[x.getnum(a) for a in fargs]
            if fname==u'geom':
                # "a"{geom(n)} gets an average of n copies of "a"
                # argument is the average number of 
                n=np.random.geometric(1/float(fargs[0]+1))-1
            elif fname=='norm':
                n=int(np.random.normal(*fargs)+.5)
            elif fname=='binom':
                n=np.random.binomial(*fargs)
            elif fname=='choose':
                n=random.choice(fargs)
            else:
                raise ValueError, 'no dist %s' % (fname)

            f=lambda lo,hi:min(hi,max(lo,n))
            args.pop()
        else:
            f=lambda lo,hi:random.randrange(lo,hi+1)
        lo=0 if len(args)==0 else x.getint(args.pop(0))
        hi=2**32 if len(args)==0 else x.getint(args.pop(0))

        #print('lo=%d hi=%d' % (lo,hi))
        n=f(lo,hi)

        s=''.join(x.sample(et) for _ in xrange(n))
        return s

    def sample(x,et):
        '''
            take a parsetree and return
        '''
        if et.data==u'alt': # choice
            return x.choose(*et.children)
        if et.data==u'cat':
            return x.concat(*et.children)
        if et.data==u'rep':
            return x.rep(et.children[0], *et.children[1].children)

        if et.data==u'range':
            lo=ord(eval(et.children[0].value))
            hi=ord(eval(et.children[1].value))
            return random.choice([chr(v) for v in range(lo,hi+1)])
        if et.data==u'string':
            return x.lit(eval(et.children[0].value))
        if et.data==u'func':
            fname=et.children[0].value
            if len(et.children)>1:
                args=et.children[1].children
            else:
                args=[]
            return getattr(x,fname)(*args)
        if et.data==u'rule':
            rname=et.children[0].value
            return x.sample(x.rule_trees[rname])
        else:
            raise GrammaParseError, '''can't transform %s''' % et
    
    def generate(x):
        while True:
            x.reset()
            del x.stack[:]
            yield x.sample(x.rule_trees['start']) 

    def reset(x):
        pass

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

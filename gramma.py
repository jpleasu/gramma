#!/usr/bin/env python
'''

    - keep around entropy source per-rule

    class GeneratePoissonRepeats(Generator):
        def __init__(self, child_generator, poisson_parameter):
            self.child=child_generator
        def gen(self):
            x=random.poisson(param)
            s=''
            for i in range(x):



def CharRangeGenerator(from,to):
    yield random.choice(charcrange)


crg=CharRangeGenerator('a','z')
crg.next()

g=Generator(crg)
g.gen()



        abcde


                r1(seed)
          r2         r3
         [0,2)      [2,5)
         ab         cde


  support(e) = { k!00, k!10, k!20 }

'''

import sys
import inspect

import random

from lark import Lark

from numpy.random import poisson

class GrammaParseError(Exception):
    pass

gramma_grammar = r"""
    ?start : ruledefs

    ruledefs : ruledef+

    ruledef: NAME ":=" alt ";"

    ?alt : cat ("|" cat)*

    ?cat : rep ("." rep)*

    ?rep: atom ( "{" rep_args "}" )?

    rep_args : INT ("," INT)? ("," FLOAT)?
            | FLOAT

    ?atom : string
         | rule
         | func
         | range
         | "(" alt ")"

    rule : NAME

    func.2 : NAME "(" args? ")"

    args : alt ("," (INT|FLOAT|alt))*

    range : "[" ESCAPED_CHAR  ".." ESCAPED_CHAR "]"

    NAME : /[a-z_][a-z_0-9]*/

    string : ESCAPED_CHAR|STRING|LONG_STRING


    STRING : /[ubf]?r?("(?!"").*?(?<!\\)(\\\\)*?"|'(?!'').*?(?<!\\)(\\\\)*?')/i
    ESCAPED_CHAR.2 : /'([^\']|\\([\nrt']|x[0-9a-fA-F][0-9a-fA-F]))'/
    LONG_STRING.2: /[ubf]?r?("(?:"").*?(?<!\\)(\\\\)*?"(?:"")|'''.*?(?<!\\)(\\\\)*?''')/is

    %import common.WS
    %import common.FLOAT
    %import common.INT

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
    parser = Lark(gramma_grammar, parser='earley', lexer='standard')


    def __init__(x, gramma_text):
        x.t=x.parser.parse(gramma_text)

        x.rule_trees={}
        for ruledef in x.t.children:
            name=ruledef.children[0].value
            expr_tree=ruledef.children[1]
            x.rule_trees[name]=expr_tree

    def getstring(x,et):
        'convert string element to python string'
        if et.data==u'string':
            l=et.children[0]
        else:
            l=et
        return eval(l.value)

    def lit(x,s):
        while True:
            yield s

    @mkgen
    def choose(x,*vals):
        return random.choice(vals)

    @mkgen
    def concat(x,*vals):
        return ''.join(vals)

    def rep(x,it,lo,hi,p):
        hi0=hi
        while True:
            if p!=None:
                hi=poisson(p)
                if hi0!=None:
                    hi=min(hi0,hi)
            else:
                hi=random.randrange(hi0-lo)

            s=''.join(it.next() for _ in xrange(lo))
            s+=''.join(it.next() for _ in xrange(hi-lo))
            yield s

    def build(x,et):
        if et.data==u'alt': # choice
            return x.choose(*[x.build(c) for c in et.children])
        if et.data==u'cat':
            return x.concat(*[x.build(c) for c in et.children])
        if et.data==u'rep':
            c=et.children[0]
            args=et.children[1].children
            hi,lo,p=None,None,None
            if len(args)==1:
                a=args[0]
                if a.type==u'INT':
                    lo=hi=int(a.value)
                elif a.type==u'FLOAT':
                    lo=0
                    p=float(a.value)
            elif len(args)==2:
                lo=int(args[0].value)
                a=args[1]
                if a.type==u'INT':
                    hi=int(a.value)
                elif a.type==u'FLOAT':
                    p=float(a.value)
                else:
                    lo=None # force fail
            elif len(args)==3:
                lo=int(args[0].value)
                hi=int(args[1].value)
                p=float(args[2].value)
    
            if lo==None:
                raise GrammaParseError, '''can't transform %s''' % et
    
            return x.rep(x.build(c),lo,hi,p)
        if et.data==u'range':
            low=ord(eval(et.children[0].value))
            hi=ord(eval(et.children[1].value))
            return x.choose(*[x.lit(chr(v)) for v in range(low,hi+1)])
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
            return x.build(x.rule_trees[rname])
        else:
            raise GrammaParseError, '''can't transform %s''' % et
    
    def generate(x):
        it=x.build(x.rule_trees['start'])
        while True:
            x.reset()
            yield it.next() 

class Example(Gramma):

    def __init__(x):
        Gramma.__init__(x,r'''
            start := words . " " . ['1'..'9'] . digit{0,5.};

            digit := ['0' .. '9'];

            words := ( "stink" | "stank" );

        ''')

def test_example():
    g=Example()
    it=g.generate()
    for i in xrange(10):
        print(it.next())


def test_parser():
    global t
    parser = Lark(gramma_grammar, parser='earley', lexer='standard')
    #    start := ( "a" | "b" ) . ['\xaa'..'\xbb'];
    t=parser.parse(r'''
        start :=  digits{1}
                | digits{.2}
                | digits{3,.4}
                | digits{5,6}
                | digits{7,8,.9}
        ;

        digits := ['1' .. '9'];

    ''')

    print(t)

if __name__ == '__main__':
    test_parser()
    #test_example()

# vim: ts=4 sw=4

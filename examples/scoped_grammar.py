#!/usr/bin/env python
'''

    demonstrate scoped variable declaraions

'''
from __future__ import absolute_import, division, print_function
from builtins import super

from gramma import *

class ScopingGrammar(GrammaGrammar):

    G=r'''
        start := "-----\n".expr;

        # indent according to 'd' statevar
        ind := "  "{`d`};

        # limit recursion by bailing after a depth (measured with 'd') of 4
        expr := `d<4`?block:ind."...";

        block := ind."{\n".                          # (indent and..) opening brace
                 push().                             # push a new context
                 (ind."def ".new(var).";\n"){1,3}.   # from 1 to 3 new variables
                 (expr."\n"){,3}.                    # recurse!
                 (ind."use ".old().";\n"){1,3}.      # from 1 to 3 "uses" of variables
                 pop().                              # pop this context
                 ind."}";                            # closing brace.

        # variable names are 1 to 5 letters, geomterically distributed w/ a mean of 3
        var := ['a'..'z']{1,5,geom(3)} ;
    '''

    def __init__(x):
        GrammaGrammar.__init__(x,type(x).G)

    def reset_state(self,state):
        state.d=0
        # list of pairs (depth, varname)
        state.vars=[]

    @gfunc
    def push(x):
        x.state.d+=1
        yield ''

    @gfunc
    def pop(x):
        x.state.d-=1
        while len(x.state.vars)>0 and x.state.vars[-1][0]>x.state.d:
            del x.state.vars[-1]
        yield ''

    @gfunc
    def new(x,ge):
        n=yield ge
        x.state.vars.append((x.state.d,n))
        yield n

    @gfunc
    def old(x):
        lvars=[v for d,v in x.state.vars if d<=x.state.d]
        yield x.random.choice(lvars)

sampler=GrammaSampler(ScopingGrammar())
print(sampler.sample())

# vim: ts=4 sw=4

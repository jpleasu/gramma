#!/usr/bin/env python3
'''

    demonstrate state cache

'''
from __future__ import absolute_import, division, print_function

import random

from gramma import *

class StateGrammar(GrammaGrammar):
    '''
         a grammar with some state gfuncs
    '''

    def __init__(self):
        GrammaGrammar.__init__(self, 'start:="";')
 
    def reset_state(self,state):
        super().reset_state(state)
        state.a=0

    @gfunc
    def incra(x):
        x.state.a+=1
        yield ''

    @gfunc
    def geta(x):
        yield '%d' % (x.state.a)



g=StateGrammar()
sampler=GrammaSampler(g)
sampler.update_statecache(a0=13)

ge=g.parse(r'''
    "initial a=". geta().  # 0 from StateGrammar.reset_state
    "\n".
    def("a",`6+3`).        # define it to the result of the python expr 6+3.. 
    "6+3=".geta().
    "\n".
    def("a",`a-4`).        # subtract 4 from it
    "a=".geta().
    "\n".
    load("a","a0").        # load it from a0, which we just set to 13..
    "a0=".geta().
    "\n".
    incra().               # increment it w/ the gfunc incra, ..
    "a0+1=".geta().
    "->a1\n".
    save("a","a1")         # save it to a1, to check after sampling.
''')

print('=======')
print('sample:')
s=sampler.sample(ge)
print(s)
print('-------')
print('after sampling, a1=%d' % sampler.get_statecache()['a1'])

print('~~~~~~~')
print('dump of metadata for gexpr:')
ge.dump_meta()




# vim: ts=4 sw=4

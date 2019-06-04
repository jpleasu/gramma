#!/usr/bin/env python
'''

    demonstrate uses of the various sampler caches

'''
from __future__ import absolute_import, division, print_function

import random

from gramma import *

from simple_grammars import ArithmeticGrammar,VarietyGrammar

#  anonymous grammar with some gcode using a param
g=GrammaGrammar('''
    start:="a"{`x`};
''',param_ids=['x'])
sampler=GrammaSampler(g)
sampler.update_params(x=7)
assert(sampler.sample()=='a'*7)
sampler.update_params(x=3)
assert(sampler.sample()=='a'*3)

# named class with a gfunc using a param
class ParamsGrammar(GrammaGrammar):
    G='''
        start:=f();
    '''
    def __init__(self):
        GrammaGrammar.__init__(self, self.G, param_ids=['x'])
    @gfunc
    def f(x):
        yield str(x.params.x)

g=ParamsGrammar()
sampler=GrammaSampler(g)
sampler.update_params(x=7)
assert(sampler.sample()=='7')
sampler.update_params(x=3)
assert(sampler.sample()=='3')


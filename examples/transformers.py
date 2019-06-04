#!/usr/bin/env python
'''

    demonstrate a transformer

'''
from __future__ import absolute_import, division, print_function

import random

from gramma import *

from simple_grammars import ArithmeticGrammar,VarietyGrammar


class LeftAltTransformer(Transformer):
    '''
        force sampler down left choice to an altdepth of maxdepth
    '''
    __slots__='maxdepth',

    def __init__(self,maxdepth=10):
        self.maxdepth=maxdepth

    def transform(self, x, ge):
        if x.state.altdepth<=self.maxdepth:
            if isinstance(ge,GAlt):
                nge=GAlt([1],[ge.children[0].copy()])
                nge.parent=ge
                return nge
        return ge

    altDepthTracker=DepthTracker(varname='altdepth', pred=lambda ge:isinstance(ge,GAlt))

g=VarietyGrammar()
sampler=GrammaSampler(g)
sampler.random.seed(0)

sampler.add_sideeffects(LeftAltTransformer.altDepthTracker)
sampler.add_transformers(LeftAltTransformer(maxdepth=30))

for i in range(10):
    s=sampler.sample()
    print('%3d %s' % (len(s),s))



# vim: ts=4 sw=4

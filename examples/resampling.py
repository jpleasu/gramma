#!/usr/bin/env python
'''
    demo resampling
'''
from __future__ import absolute_import, division, print_function

import random

from datetime import datetime,timedelta

from gramma2 import *

from simple_grammars import ArithmeticGrammar

g=ArithmeticGrammar()
sampler=GrammaSampler(g)
tracer=Tracer()
sampler.add_sideeffects(tracer)

origs=sampler.sample()
tt=tracer.tracetree
print('-- the original sample --')
print(origs)

## choose the node to resample, n
n=random.choice([n for n in tt.gennodes() if isinstance(n.ge,GRule)])

print('-- resampling --')
print('"%s" at depth(n) = %d' % (n.ge,n.depth()))

## construct a GExpr that resamples only n
rge,cfg=tt.resample(g,lambda t:t==n)
print('-- the resample expression --')
print(rge)

sampler.update_cache(cfg)

for i in range(10):
    print('---')
    print(sampler.sample(rge))



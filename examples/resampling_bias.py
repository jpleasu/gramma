#!/usr/bin/env python
'''
    demo resampling with a bias to the previous sample
'''
from __future__ import absolute_import, division, print_function

import random

from datetime import datetime,timedelta

from gramma import *

from simple_grammars import ArithmeticGrammar,VarietyGrammar

from collections import Counter

def map_subs(function, sequence, *replacements):
    '''
        like map, with replacements
    '''
    m=defaultdict(function) 
    for rk,rv in zip(replacements,replacements[1:]):
        m[rk]=rv
    return [m[x] for x in sequence]


class Resampler(object):
    def __init__(self,factorm=.01,factore=1.1):
        '''
            odds of choosing random over prevous behavior is
                factor = m * depth ** e
        '''
        self.cfg=None
        self._factorm=factorm
        self._factore=factore

    @property
    def factor(self):
        return self._factorm*(self._factore)**self.d
    
    def recurse_GRule(self,tt):
        return self.recurse(tt.children[0])

    def recurse_GAlt(self,tt):
        return GAlt([self.factor,1], [tt.ge.copy(), self.recurse(tt.children[0])])
        #return self.recurse(tt.children[0])

    def recurse_GTern(self,tt):
        return GAlt([self.factor,1], [tt.ge.copy(), self.recurse(tt.children[0])])

    def recurse_GCat(self,tt):
        return GCat([self.recurse(c) for c in tt.children])

    def recurse_GRep(self,tt):
        rge=GCat([self.recurse(c) for c in tt.children])
        return GAlt([self.factor,1],[tt.ge.copy(),rge])

    def recurse_default(self,tt):
        return GAlt([self.factor,1],[tt.ge.copy(),GTok.from_str(tt.s)])

    def recurse(self, tt):
        self.d+=1
        ge= getattr(self,'recurse_%s' % tt.ge.__class__.__name__, self.recurse_default)(tt)
        self.d-=1
        return ge

    def __call__(self,tt):
        self.d=0
        self.cfg=CacheConfig()
        ge=self.recurse(tt)
        return self.cfg,ge.simplify()

g=ArithmeticGrammar()
#g=VarietyGrammar()
sampler=GrammaSampler(g)
tracer=Tracer()
sampler.add_sideeffects(tracer)

origs=sampler.sample()
tt=tracer.tracetree
print('-- the original sample --')
print(origs)


N=20
print('-- tuning parameters for 1 in %d repeats of the original --' % N)

resampler=Resampler(.001,1.01)
while True:
    cfg,rge=resampler(tt)
    sampler.update_cache(cfg)
    cnt=Counter(islice(sampler.gensamples(rge),N))
    c=cnt[origs]
    if c==1:
        break
    #print(c)
    if c==0:
        resampler._factorm*=.9
        resampler._factore*=.9
    else:
        resampler._factorm*=1.001
        resampler._factore*=1.01

print('m=%f e=%f'  %(resampler._factorm, resampler._factore))    
#print('-- the resample expression --')
#print(rge)

print('-- resampling --')

for i in range(10):
    print('---')
    s=sampler.sample(rge)
    if s==origs:
        print(' ORIG ')
    else:
        print(s)



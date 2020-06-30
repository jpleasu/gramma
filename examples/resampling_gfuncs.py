#!/usr/bin/env python3
"""
    demo resampling when more complicated gfuncs are present
"""
from __future__ import absolute_import, division, print_function

import random

from gramma import *


class FuncyGrammar(GrammaGrammar):
    G = r'''
        start := r;
        r :=  "s."
            | `1-revcalled` "rev(".rev(r).")"
            | decide("t"|"f","a.".r,"b.".r) 
            | dub("a"|"b")
            | "c." . r;
    '''

    def __init__(self):
        GrammaGrammar.__init__(self, type(self).G, sideeffects=[DepthTracker])

    def reset_state(self, state):
        state.revcalled = False

    @gfunc
    def rev(x, child):
        x.state.revcalled = True
        yield ''.join(reversed((yield child)))

    @gfunc
    def decide(x, child1, child2, child3):
        s1 = yield child1
        if 't' in s1:
            yield (yield child2)
        yield (yield child3)

    @gfunc
    def dub(x, child):
        yield (yield child) + (yield child)


g = FuncyGrammar()
sampler = GrammaSampler(g)
tracer = Tracer()
sampler.add_sideeffects(tracer)

# generate a sample w/ at least one gfunc in its trace
while True:
    origs = sampler.sample()
    tt = tracer.tracetree
    funcs = [n for n in tt.gennodes() if isinstance(n.ge, GFunc)]
    if len(funcs) > 0:
        break
f = random.choice(funcs)
n = random.choice([n for n in f.gennodes() if not isinstance(n.ge, GFunc) and n.ge.get_meta().uses_random])
print('-- the original sample --')
print(origs)
print('  with %d func node(s)' % len(funcs))
print('chose func "%s" at depth(n) = %d' % (f.ge.fname, f.depth()))
print(' and child %s at depth(n) = %d' % (n.ge, n.depth()))
# tt.dump()

## construct a GExpr that resamples only n
rge, cfg = tt.resample(g, lambda t: t == n)
print('-- the resample expression --')
print(rge)

sampler.update_cache(cfg)

for i in range(10):
    print('---')
    print(sampler.sample(rge))

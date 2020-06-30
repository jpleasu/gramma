#!/usr/bin/env python3
"""

    demonstrate uses of random cache

"""
from __future__ import absolute_import, division, print_function

from simple_grammars import ArithmeticGrammar, VarietyGrammar

from gramma import *

g = VarietyGrammar()
sampler = GrammaSampler(g)


# sampler.random.seed(0)

def p(n, e):
    s = sampler.sample(e)
    print('---- %s ----' % n)
    print('%3d %s' % (len(s), s))
    return s


# generate a sample and save the random state on entry
a = p('A', 'save_rand("r0").start')
# generate a new sample, demonstrating a different random state
b = p('B', 'start')

# resume at r0 again:
a1 = p('A', 'load_rand("r0").start')
assert (a1 == a)
# if we generate a new sample here, the state resumes from r0 again, so it will be A
b1 = p('B', 'start')
assert (b1 == b)

# so if we want to resume at r0..
a1 = p('A', 'load_rand("r0").start')
assert (a1 == a)
# .. and continue w/ a new random, we need to reseed:
sampler.random.seed(None)  # none draws a new seed from urandom
c = p('C', 'start')

# and we can still resume from r0 later.
a1 = p('A', 'load_rand("r0").start')
assert (a1 == a)
# we can also reseed the random number generator from within the grammar
d = p('D', 'reseed_rand().start')

## resample with a cached randstate.

g = ArithmeticGrammar()
sampler = GrammaSampler(g)
tracer = Tracer()
sampler.add_sideeffects(tracer)
origs = sampler.sample()

tt = tracer.tracetree
# Note that the dynamic alternations used in ArithmeticGrammar use depth
# and using "cat(load_rand,start)" increases the depth. Reset the depth
# with the 'def'. (depth is set to 1 because on _exit_ from the gfunc
# call, depth is decremented)
r = tt.first(lambda n: n.ge.get_meta().uses_random).inrand
sampler.random.set_cached_state('r', r)
s = sampler.sample('def("depth",1).load_rand("r").start')
assert (s == origs)

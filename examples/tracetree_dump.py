#!/usr/bin/env python
'''

    demonstrate use of the Tracer to generate a tracetree

'''
from __future__ import absolute_import, division, print_function

import random

from gramma2 import *

from simple_grammars import ArithmeticGrammar

#g=VarietyGrammar()
g=ArithmeticGrammar()

tracing_sampler=GrammaSampler(g)
tracer=Tracer()
tracing_sampler.add_sideeffects(tracer)

#tracing_sampler.random.seed(79)
tracing_sampler.random.seed(319)

s=tracing_sampler.sample()
print('a fixed sample:')
print(s)
print()
print('its tracetree:')
tracer.tracetree.dump()



print('now we sample again from a captured random state...')
# as long as we get the state from a sampler, we can always reconstruct a sample's tracetree later...
sampler=GrammaSampler(g)

# sample a few times, then save state, sample some more, the revisit our 3rd
# sample with a tracing sampler

s=sampler.sample()
s=sampler.sample()

# remember randome state on entering third sample.
randstate_of_third_sample=sampler.random.get_state()
s3=sampler.sample()

s=sampler.sample()
s=sampler.sample()
s=sampler.sample()

# verify we get the same thing w/ the tracing sampler after reusing the saved
# random state:
tracing_sampler.random.set_state(randstate_of_third_sample)
s3x=tracing_sampler.sample()
assert(s3==s3x)

# demo finding the node that contains a byte of the sample
print(".. and let's find the node that generated the character at some random position")
print(s3x)
i=random.randrange(0,len(s3x))
print(' '*i + '''^- here''')
print('looking up the tracetree from the leaf we see')
n=tracer.tracetree.child_containing(i)
d=0
while n!=None:
    print('   %s%s' % (' '*d, n.ge))
    n=n.parent
    d+=1
#n.dump()
#print(n.ge)




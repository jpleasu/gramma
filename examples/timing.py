#!/usr/bin/env python
'''

    demonstrate use of the Tracer to generate a tracetree

'''
from __future__ import absolute_import, division, print_function

import random

from gramma2 import *

from simple_grammars import ArithmeticGrammar



def demo_timing():
    import time

    g=ArithmeticGrammar()
    sampler=GrammaSampler(g)
    sampler.random.seed(0)
    n=2500
    #n=10000
    t0=perf_counter()
    for i in xrange(n):
        s=sampler.sample()
    t1=perf_counter()
    print('avg=%f' % ((t1-t0)/n))

def demo_profile():
    import cProfile
    cProfile.run('demo_timing()')

demo_timing()

#!/usr/bin/env python3
'''

    demonstrate use of the Tracer to generate a tracetree

'''
from __future__ import absolute_import, division, print_function

import random

from gramma import *

from simple_grammars import ArithmeticGrammar

import timeit

import time

def demo_timing1():
    g=ArithmeticGrammar()
    sampler=GrammaSampler(g)
    sampler.random.seed(0)
    n=2500
    #n=10000
    t0=perf_counter()
    #for i in xrange(n):
    #    s=sampler.sample()
    for s in islice(sampler.gensamples(),n):
        pass
    t1=perf_counter()
    print('avg=%f' % ((t1-t0)/n))

def demo_timing2():
    sampler.random.seed(0)
    for i in range(100):
        sampler.sample()

def init():
    global g, sampler
    g=ArithmeticGrammar()
    sampler=GrammaSampler(g)

def demo_profile():
    import cProfile
    cProfile.run('demo_timing1()')



if __name__=='__main__':
    #print(timeit.timeit("demo_timing2()", setup="from __main__ import init,demo_timing2;init()"))
    demo_timing1()
    #demo_profile()

#!/usr/bin/env python
'''
    recursion depth can be controlled in several ways
'''
from __future__ import absolute_import, division, print_function

import random

from gramma import *


# define a depth tracker for use later
ruleDepthTracker=RuleDepthTracker()

# we can use a weighted alternation to control recursion depth
g=GrammaGrammar('start :=  (3 "" | 10 "a".start );')
sampler=GrammaSampler(g)
s=sampler.sample()
print(s)



# we can use a ternary operator and a DepthTracker to stop at a fixed depth
g=GrammaGrammar('start :=  `depth<=3` ? "a" . start : "" ;', sideeffects=[ruleDepthTracker])
sampler=GrammaSampler(g)
s=sampler.sample()
print(s)


# we can use a dynamically weighted alternation with a depth tracker
g=GrammaGrammar('start :=  `30 if depth>=10 else 1` "" |  10 "a" . start ;', sideeffects=[ruleDepthTracker])
sampler=GrammaSampler(g)
s=sampler.sample()
print(s)


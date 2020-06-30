#!/usr/bin/env python3
"""

using the def gfunc

"""

from __future__ import absolute_import, division, print_function

from gramma import *


class UsingDef(GrammaGrammar):
    G = r'''
        start := r;

        r:=def('x',`x+1`).(
              `print(x) or x/(1.+x)` "a" | "b".r 
        ).def('x',`x-1`);

    '''

    def __init__(self):
        GrammaGrammar.__init__(self, type(self).G, sideeffects=[DepthTracker])

    def reset_state(self, state):
        state.x = 0


if __name__ == '__main__':
    print(GrammaSampler(UsingDef()).sample())

# vim: ts=4 sw=4

#!/usr/bin/env python3

from gramma.samplers import GrammaInterpreter, gfunc

import os

GLF = __file__[:-3] + '.glf'


class VarietySampler(GrammaInterpreter):
    a: bool

    def __init__(self):
        super().__init__(open(GLF))
        self.rule_depth = 0
        self.a = False

    @gfunc
    def f(self):
        self.a = not self.a
        return self.create_sample('f')

    @gfunc
    def ff(self):
        return self.create_sample('ff')

    def a_func(self):
        return .1


if __name__ == '__main__':
    s = VarietySampler()
    for i in range(10):
        samp = s.coro_sample_start()
        #samp = s.sample_start()
        print(samp.s)

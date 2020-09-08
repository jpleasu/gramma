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
        return 2 if self.a else 1


if __name__ == '__main__':
    import sys
    s = VarietySampler()
    while True:
        samp = s.coro_sample_start()
        sys.stdout.write(samp.s)

#!/usr/bin/env python3

from typing import Any

from gramma.samplers import GrammaInterpreter, gfunc, gdfunc, Sample


class SMTSampler(GrammaInterpreter):

    def __init__(self):
        super().__init__(open(__file__[:-3]+'.glf'))
        self.sort_rec = .1
        self.array_sexpr_rec = .001

    @gdfunc
    def mk_array_sort(self, domain, range):
        return 'array', domain, range

    @gfunc
    def domain(self, a):
        return a.d[1]

    @gfunc
    def range(self, a):
        return a.d[2]

    @gfunc(lazy=True)
    def switch_sort(self, sort, i, b, a):
        d = self.sample(sort).d
        if d == 'i':
            return self.sample(i)
        elif d == 'b':
            return self.sample(b)
        return self.sample(a)

    @gfunc(lazy=True)
    def coro_switch_sort(self, sort, i, b, a):
        d = (yield sort).d
        if d == 'i':
            yield (yield i)
        elif d == 'b':
            yield (yield b)
        yield (yield a)

    def denote(self, s: Sample, d: Any):
        return Sample(s.s, d)



if __name__ == '__main__':
    import sys

    sys.setrecursionlimit(10000)
    s = SMTSampler()
    s.random.seed(1)
    for i in range(100):
        samp = s.sample_start()
        # samp = s.coro_sample_start()
        print(samp.s)

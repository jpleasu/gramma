#!/usr/bin/env python3

from typing import Any, Union, Literal, Tuple, Generator, TypeVar, Optional

from gramma.parser import GExpr
from gramma.samplers import GrammaInterpreter, gfunc, gdfunc, Sample
from gramma.samplers.interpreter import GrammaSamplerError


class Denotation:
    type_: Literal['i', 'b', 'array']
    d: Optional['SampleT']
    r: Optional['SampleT']

    def __init__(self, type_: Literal['i', 'b', 'array'], domain: Optional['SampleT'] = None,
                 range_: Optional['SampleT'] = None):
        self.type_ = type_
        self.d = domain
        self.r = range_

    @property
    def domain(self) -> 'SampleT':
        if self.d is None:
            raise GrammaSamplerError('array sort missing domain')
        return self.d

    @property
    def range_(self) -> 'SampleT':
        if self.r is None:
            raise GrammaSamplerError('array sort missing range')
        return self.r


class SampleT(Sample[Denotation]):
    @property
    def den(self) -> Denotation:
        if self.d is None:
            raise GrammaSamplerError('sample missing expected denotation')
        return self.d


SampleFactory = Generator[Union[GExpr, SampleT], SampleT, None]


class SMTSampler(GrammaInterpreter[SampleT, Denotation]):

    def __init__(self):
        super().__init__(open(__file__[:-3] + '.glf'))
        self.sort_rec = .1
        self.array_sexpr_rec = .001

    @staticmethod
    def create_sample(*args: Any, **kwargs: Any) -> SampleT:
        return SampleT(*args, **kwargs)

    def denote(self, samp: SampleT, d: Denotation) -> SampleT:
        return self.create_sample(samp.s, d)

    @gdfunc
    def mk_array_sort(self, domain: SampleT, range_: SampleT) -> Denotation:
        return Denotation('array', domain, range_)

    @gfunc
    def domain(self, a: SampleT) -> SampleT:
        return a.den.domain

    @gfunc(fname='range')
    def range_(self, a: SampleT) -> SampleT:
        return a.den.range_

    @gfunc(lazy=True)
    def switch_sort(self, sort: GExpr, i: GExpr, b: GExpr, a: GExpr) -> SampleT:
        d = self.sample(sort).d
        if d == 'i':
            return self.sample(i)
        elif d == 'b':
            return self.sample(b)
        return self.sample(a)

    @gfunc(lazy=True)
    def coro_switch_sort(self, sort: GExpr, i: GExpr, b: GExpr, a: GExpr) -> SampleFactory:
        d = (yield sort).d
        if d == 'i':
            yield (yield i)
        elif d == 'b':
            yield (yield b)
        yield (yield a)


if __name__ == '__main__':
    import sys

    sys.setrecursionlimit(10000)
    s = SMTSampler()
    # s.random.seed(1)
    while True:
        sample = s.sample_start()
        # sample = s.coro_sample_start()
        sys.stdout.write(sample.s)

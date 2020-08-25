#!/usr/bin/env python3

from typing import Any

from gramma.samplers import GrammaInterpreter, gfunc, gdfunc, Sample


class SMTSampler(GrammaInterpreter):
    GLF = '''\
        start := equals(sort);
        equals(s) := '(= '.sexpr(s).' '.sexpr(s).' )';


        #### sexprs ####
        sexpr(s) := switch_sort(s, int_sexpr, bool_sexpr, array_sexpr(domain(s), range(s)));

        bool_sexpr := ('true' | 'false') / 'b'
                    | array_wrap(bool_sort)
                    ;
        int_sexpr := ['1'..'9'].['0'..'9']{geom(4)} / 'i'
                    | array_wrap(int_sort)
                    ;
        array_sexpr(domain_sort, range_sort) :=
                      "((as const (Array ".domain_sort." ".range_sort.")) ".sexpr(range_sort).")"
                    | `array_sexpr_rec` "(store ".array_sexpr(domain_sort, range_sort)." ".sexpr(domain_sort)." ".sexpr(range_sort).")"
                    ;
        const_array_sexpr :=
                    '(store (store (store ((as const (Array Int Int)) 0) 0 1) 1 2) 0 0)';

        # generate an sexpr of the given type by invoking an array whose range is the given type
        array_wrap(s) := choose domain~sort in
                    "( select " . array_sexpr(domain, s)." ".sexpr(domain).")";



        #### sorts ####

        int_sort := 'Int'/'i';
        bool_sort := 'Bool'/'b';
        array_sort(d, r) := '( Array '.d.' '.r.' )' / mk_array_sort(d,r);

        # random sort
        sort := int_sort | bool_sort | `sort_rec` array_sort(sort,sort);

    '''

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

    def __init__(self, glf=GLF):
        super().__init__(glf)
        self.sort_rec = .1
        self.array_sexpr_rec = .001

    def denote(self, s: Sample, d: Any):
        return Sample(s.s, d)


if __name__ == '__main__':
    import sys

    sys.setrecursionlimit(10000)
    s = SMTSampler()
    s.random.seed(1)
    for i in range(100):
        samp = s.sample_start()
        print(samp.s)

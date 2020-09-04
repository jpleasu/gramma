#!/usr/bin/env python3

import unittest
from functools import reduce
from typing import Any

from gramma.parser import GExpr
from gramma.samplers import GrammaInterpreter, gfunc, gdfunc, Sample


class Arithmetic(GrammaInterpreter):
    GLF = '''
        start := expr;
        expr := add;
        add := mul . ('+'.mul){,3};
        mul := atom . ('*'.atom){,3};
        atom :=   'x'
                | randint()
                | `expr_rec` '(' . expr . ')';
    '''

    def __init__(self, glf=GLF):
        super().__init__(glf)
        self.expr_rec = .1

    @gfunc
    def randint(self):
        return self.create_sample(str(self.random.integers(0, 100000)))


def flatlist(a):
    if a is None:
        return []
    elif isinstance(a, list):
        return a
    else:
        return [a]


class SemanticArithmetic(Arithmetic):
    """
        denotations are
            null for unadorned tokens
            integers for rules
            list of integers otherwise
    """
    GLF = '''
        start := expr;
        expr := add;
        add := mul . ('+'.mul){,3} / 'sum';
        mul := atom . ('*'.atom){,3} / 'product';
        atom :=   'x' / 'variable'
                | randint()
                | `expr_rec` '(' . expr . ')';
    '''

    def __init__(self, glf=GLF):
        super().__init__(glf)
        self.expr_rec = .1
        self.variables = dict()

    @staticmethod
    def cat(a: Sample, b: Sample) -> Sample:
        return Sample(a.s + b.s, flatlist(a.d) + flatlist(b.d))

    def denote(self, a: Sample, b: Any) -> Sample:
        if b == 'variable':
            d = self.variables.get(a.s)
        elif b == 'sum':
            d = sum(a.d)
        elif b == 'product':
            d = reduce(lambda x, y: x * y, a.d)
        return Sample(a.s, d)

    @gfunc
    def randint(self):
        n = self.random.integers(0, 100000)
        return Sample(str(n), n)


class TestInterpreter(unittest.TestCase):

    def test_tok(self):
        s = GrammaInterpreter('''
            start := 'abc';
        ''')
        self.assertEqual(str(s.sample_start()), 'abc')

    def test_cat(self):
        s = GrammaInterpreter('''
            start := 'a' . 'b' . 'c';
        ''')
        self.assertEqual(str(s.sample_start()), 'abc')

    def test_alt(self):
        s = GrammaInterpreter('''
            start := 'a' | 'b' | 'c';
        ''')
        s.random.seed(1)
        self.assertEqual(''.join(str(s.sample_start()) for i in range(10)), 'acbbacbbaa')

    def test_alt_numerical_weights(self):
        s = GrammaInterpreter('''
            start := 1.2 'a' | 2 'b' | 'c';
        ''')
        s.random.seed(1)
        self.assertEqual(''.join(str(s.sample_start()) for i in range(10)), 'abbbacbbaa')

    def test_alt_gcode_weights(self):
        s = GrammaInterpreter('''
            start := 'a' | `False` 'b' | 'c';
        ''')
        s.random.seed(1)
        self.assertEqual(''.join(str(s.sample_start()) for i in range(10)), 'acccacaaaa')

    def test_rules_simple(self):
        s = GrammaInterpreter('''
            start := r1 . r2 . r3;
            r1 := 'd';
            r2 := 'e';
            r3 := 'f';
        ''')
        self.assertEqual(str(s.sample_start()), 'def')

    def test_choosein(self):
        s = GrammaInterpreter('''
            start := choose x ~ 'a' in x.x.x;
        ''')
        self.assertEqual(str(s.sample_start()), 'aaa')

    def test_choosein2(self):
        s = GrammaInterpreter('''
            start := choose x ~ 'a'|'b', y ~ 'c'|'d' in x.x.x.y.y.y;
        ''')
        s.random.seed(1)
        self.assertEqual(str(s.sample_start()), 'aaaddd')

    def test_choosein3(self):
        s = GrammaInterpreter('''
            start := choose x ~ 'a' in x.( choose x ~ 'b' in x).x;
        ''')
        s.random.seed(1)
        self.assertEqual(str(s.sample_start()), 'aba')

    def test_denoted_gcode(self):
        s = GrammaInterpreter('''
            start := 'a'/`17`;
        ''')
        self.assertEqual(s.sample_start().d, 17)

    def test_gfunc_lazy(self):
        class G(GrammaInterpreter):
            @gfunc(lazy=True)
            def f(self, ge: GExpr) -> Sample:
                return self.cat(self.sample(ge), self.sample(ge))

        s = G('''
            start := f('a'|'b');
        ''')
        s.random.seed(1)

        x = s.sample_start()
        self.assertEqual(x.s, 'ab')

    def test_denoted_dfunc(self):
        class G(GrammaInterpreter):
            @gdfunc
            def f(self):
                return 17

        s = G('''
            start := 'a'/f();
        ''')
        x = s.sample_start()
        self.assertEqual(x.d, 17)

    def test_denoted_choosein(self):
        class G(GrammaInterpreter):
            @gdfunc
            def f(self, x: Sample) -> str:
                return x.s

        s = G('''
            start := choose x~'somevar' in 'a'/f(x);
        ''')
        x = s.sample_start()
        self.assertEqual(x.d, 'somevar')

    def test_rules_parameterized(self):
        s = GrammaInterpreter('''
            start := r('a'). ',' .r('b');
            r(x) := x . x;
        ''')
        self.assertEqual(str(s.sample_start()), 'aa,bb')

    def test_rules_parameterized2(self):
        s = GrammaInterpreter('''
            start := r1('a'). ',' .r2('b');
            r1(x) := x . x;
            r2(x) := r1(x) . '-' . r1(r1(x));
        ''')
        self.assertEqual(str(s.sample_start()), 'aa,bb-bbbb')

    def test_rules_parameterized3(self):
        s = GrammaInterpreter('''
            start := r('a'|'b'). ',' .r('c'|'d');
            r(x) := x {5};
        ''')
        s.random.seed(1)
        self.assertEqual(str(s.sample_start()), 'aaaaa,ddddd')

    def test_rules_parameterized4(self):
        s = GrammaInterpreter('''
            start := r('a'|'b', 'c'|'d');
            r(x,y) := x.y;
        ''')
        s.random.seed(1)
        self.assertEqual(str(s.sample_start()), 'ad')

    def test_rep1(self):
        s = GrammaInterpreter('''
            start := 'a'{3};
        ''')
        self.assertEqual(str(s.sample_start()), 'aaa')

    def test_rep2(self):
        s = GrammaInterpreter('''
            start := 'a'{3,4};
        ''')
        s.random.seed(1)
        self.assertEqual(','.join(str(s.sample_start()) for i in range(10)),
                         'aaa,aaaa,aaaa,aaa,aaaa,aaaa,aaaa,aaa,aaa,aaa')

    def test_rep_dists(self):
        s = GrammaInterpreter('''
               start := 'a'{geom(3)};
           ''')
        s.random.seed(1)
        self.assertEqual(','.join(str(s.sample_start()) for i in range(10)),
                         ',aaaa,aa,aa,,aaaaaaaa,aa,a,,a')

        s = GrammaInterpreter('''
               start := 'a'{norm(5,3)};
           ''')
        s.random.seed(1)
        self.assertEqual(','.join(str(s.sample_start()) for i in range(10)),
                         'aaaaaaaaaaa,aa,aaaaaaa,aaaaaa,aaaa,aaaa,aaaaaaaaaaa,aaaa,aaaaaa,aaaa')

        s = GrammaInterpreter('''
               start := 'a'{binom(5,.7)};
           ''')
        s.random.seed(1)
        self.assertEqual(','.join(str(s.sample_start()) for i in range(10)),
                         'aaaa,aaa,aaa,aaaa,aaaaa,aa,aaaa,aaaa,aaaa,aaaa')

        s = GrammaInterpreter('''
               start := 'a'{choice(1,2,3)};
           ''')
        s.random.seed(1)
        self.assertEqual(','.join(str(s.sample_start()) for i in range(10)),
                         'a,aaa,aa,aa,a,aaa,aa,aa,a,a')

    def test_rep_gcode(self):
        class G(GrammaInterpreter):
            x: int

        s = G('''
               start := 'a'{`x-1`, `x+1`};
           ''')
        s.x = 3

        s.random.seed(1)
        self.assertEqual(','.join(str(s.sample_start()) for i in range(10)), 'aa,aaaa,aaaa,aa,aaa,aaaa,aaa,aa,aa,aa')

    def test_grange(self):
        s = GrammaInterpreter('''
               start := ['a'..'z'];
           ''')
        s.random.seed(1)
        self.assertEqual(''.join(str(s.sample_start()) for i in range(10)), 'gxscoynedi')

    def test_basic_operators(self):
        s = GrammaInterpreter('''
            start := r1 . ','. r2{2,3} . ',' . r3;
            r1 :=  'a' | 'b';
            r2 :=  'c' | 'd';
            r3 :=  'e' | 'f';
        ''')
        s.random.seed(1)
        self.assertEqual(str(s.sample_start()), 'a,cdc,e')

    def test_arithmetic_grammar(self):
        s = Arithmetic()
        s.random.seed(1)
        self.assertEqual(str(s.sample_start()), 'x*x*17781')
        self.assertEqual(str(s.sample_start()), 'x+15135')
        self.assertEqual(str(s.sample_start()), '2810*35917*x')
        self.assertEqual(str(s.sample_start()), '99487*x*89714')
        self.assertEqual(str(s.sample_start()), 'x*94935+x*x*x')
        self.assertEqual(str(s.sample_start()), '77305*x*70020+x*10991*85623*'
                                                '(x*x*(x*6795*x*30102+x*x*x)*18804+x*33287*18412*x+x*x+x*x*x*x)'
                                                '+x*50515*x*x+x')

    def test_semantic_arithmetic_grammar(self):
        s = SemanticArithmetic()
        s.variables['x'] = 2

        s.random.seed(1)
        samp = s.sample_start()
        self.assertEqual(samp.s, 'x*x*17781')
        self.assertEqual(samp.d, 71124)

        samp = s.sample_start()
        self.assertEqual(samp.s, 'x+15135')
        self.assertEqual(samp.d, 15137)


class TestCoroInterpreter(unittest.TestCase):
    def test_basic(self):
        s = GrammaInterpreter('''
            start := r1 . ','. r2{2,3} . ',' . r3;
            r1 :=  'a' | 'b';
            r2 :=  'c' | 'd';
            r3 :=  'e' | 'f';
        ''')
        s.random.seed(1)
        samp = s.coro_sample_start()
        self.assertEqual(str(samp), 'a,cdc,e')

    def test_arithmetic_grammar(self):
        s = Arithmetic()
        s.random.seed(1)
        self.assertEqual(str(s.coro_sample_start()), 'x*x*17781')
        self.assertEqual(str(s.coro_sample_start()), 'x+15135')
        self.assertEqual(str(s.coro_sample_start()), '2810*35917*x')
        self.assertEqual(str(s.coro_sample_start()), '99487*x*89714')
        self.assertEqual(str(s.coro_sample_start()), 'x*94935+x*x*x')
        self.assertEqual(str(s.coro_sample_start()), '77305*x*70020+x*10991*85623*'
                                                     '(x*x*(x*6795*x*30102+x*x*x)*18804+x*33287*18412*x+x*x+x*x*x*x)'
                                                     '+x*50515*x*x+x')

    def test_semantic_arithmetic_grammar(self):
        s = SemanticArithmetic()
        s.variables['x'] = 2

        s.random.seed(1)
        samp = s.coro_sample_start()
        self.assertEqual(samp.s, 'x*x*17781')
        self.assertEqual(samp.d, 71124)

        samp = s.coro_sample_start()
        self.assertEqual(samp.s, 'x+15135')
        self.assertEqual(samp.d, 15137)


if __name__ == '__main__':
    unittest.main()

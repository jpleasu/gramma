#!/usr/bin/env python3

import unittest
from functools import reduce
from typing import Any, Callable, Protocol

from gramma.parser import GExpr
from gramma.samplers import GrammaInterpreter, gfunc, gdfunc, Sample
from gramma.samplers.interpreter import GrammaSamplerError, OperatorsImplementationSamplerMixin


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


class GFake(GExpr):
    pass


class SampleStartFunc(Protocol):
    def __call__(self, __a: GrammaInterpreter) -> Sample:
        ...


class SampleFunc(Protocol):
    def __call__(self, __a: GrammaInterpreter, __b: GExpr) -> Sample:
        ...


class TestExceptionsBase(unittest.TestCase):
    sample: SampleFunc
    sample_start: SampleStartFunc

    def test_sample(self):
        s = GrammaInterpreter('''
            start := 'abc';
        ''')
        with self.assertRaises(GrammaSamplerError):
            self.sample(s, GFake())

    def test_RepDist(self):
        s = GrammaInterpreter('''
            start := 'abc'{fake(1,2,3)};
        ''')
        with self.assertRaises(GrammaSamplerError):
            self.sample_start(s)

    def test_GRule_wrong_args(self):
        s = GrammaInterpreter('''
            start := r('a');
            r(a,b) := a.b;
        ''')
        with self.assertRaises(GrammaSamplerError):
            self.sample_start(s)

    def test_GDFunc_missing(self):
        with self.assertRaises(GrammaSamplerError):
            s = GrammaInterpreter('''
                start := 'a'/missing();
            ''')
            self.sample_start(s)


class TestExceptions(TestExceptionsBase):
    def setUp(self) -> None:
        self.sample = GrammaInterpreter.sample
        self.sample_start = GrammaInterpreter.sample_start


class TestCoroExceptions(TestExceptionsBase):
    def setUp(self) -> None:
        self.sample = GrammaInterpreter.coro_sample
        self.sample_start = GrammaInterpreter.coro_sample_start


del TestExceptionsBase


class BaseTestInterpreter(unittest.TestCase):
    sample_start: SampleStartFunc

    def test_GTok(self):
        s = GrammaInterpreter('''
            start := 'abc';
        ''')
        self.assertEqual(str(self.sample_start(s)), 'abc')

    def test_Gcat(self):
        s = GrammaInterpreter('''
            start := 'a' . 'b' . 'c';
        ''')
        self.assertEqual(str(self.sample_start(s)), 'abc')

    def test_GAlt(self):
        s = GrammaInterpreter('''
            start := 'a' | 'b' | 'c';
        ''')
        s.random.seed(1)
        self.assertEqual(''.join(str(self.sample_start(s)) for i in range(10)), 'acbbacbbaa')

    def test_GAlt_numerical_weights(self):
        s = GrammaInterpreter('''
            start := 1.2 'a' | 2 'b' | 'c';
        ''')
        s.random.seed(1)
        self.assertEqual(''.join(str(self.sample_start(s)) for i in range(10)), 'abbbacbbaa')

    def test_GAlt_gcode_weights(self):
        s = GrammaInterpreter('''
            start := 'a' | `False` 'b' | 'c';
        ''')
        s.random.seed(1)
        self.assertEqual(''.join(str(self.sample_start(s)) for i in range(10)), 'acccacaaaa')

    def test_GTern(self):
        s = GrammaInterpreter('''
            start := `random.binomial(1,.5)`?'a':'b';
        ''')
        s.random.seed(1)
        self.assertEqual(''.join(str(self.sample_start(s)) for i in range(10)), 'baaababbbb')

    def test_GRule_simple(self):
        s = GrammaInterpreter('''
            start := r1 . r2 . r3;
            r1 := 'd';
            r2 := 'e';
            r3 := 'f';
        ''')
        self.assertEqual(str(self.sample_start(s)), 'def')

    def test_GChooseIn(self):
        s = GrammaInterpreter('''
            start := choose x ~ 'a' in x.x.x;
        ''')
        self.assertEqual(str(self.sample_start(s)), 'aaa')

    def test_GChooseIn_multiple(self):
        s = GrammaInterpreter('''
            start := choose x ~ 'a'|'b', y ~ 'c'|'d' in x.x.x.y.y.y;
        ''')
        s.random.seed(1)
        self.assertEqual(str(self.sample_start(s)), 'aaaddd')

    def test_GChooseIn_nested(self):
        s = GrammaInterpreter('''
            start := choose x ~ 'a' in x.( choose x ~ 'b' in x).x;
        ''')
        s.random.seed(1)
        self.assertEqual(str(self.sample_start(s)), 'aba')

    def test_GDenoted_gcode(self):
        s = GrammaInterpreter('''
            start := 'a'/`17`;
        ''')
        self.assertEqual(self.sample_start(s).d, 17)

    def test_GFunc_lazy(self):
        class G(GrammaInterpreter):
            @gfunc(lazy=True)
            def f(self, ge: GExpr) -> Sample:
                return self.cat(self.sample(ge), self.sample(ge))

        s = G('''
            start := f('a'|'b');
        ''')
        s.random.seed(1)
        x = self.sample_start(s)
        self.assertEqual(x.s, 'ab')

    def test_GCode(self):
        class G(GrammaInterpreter):
            @gfunc
            def f(self, ns: Sample) -> Sample:
                n = int(ns.s)
                return Sample(str(2 * n))

            @gfunc
            def g(self, ns: Sample) -> Sample:
                return ns

        s = G('''
            start := f(`1+1`).','.g(`Sample('a')`);
        ''')
        s.random.seed(1)
        self.assertEqual(str(self.sample_start(s)), '4,a')

    def test_GCode_variable_update(self):
        class G(GrammaInterpreter):
            def __init__(self, glf):
                super().__init__(glf)
                self.x = 1

            @gfunc
            def f(self, ns: Sample) -> Sample:
                return ns

        s = G('''
            start := f(`_(x,x=7)`).','.f(`_(x,x=30)`);
        ''')
        s.random.seed(1)
        self.assertEqual(str(self.sample_start(s)), '1,7')

    def test_GDenoted_dfunc(self):
        class G(GrammaInterpreter):
            @gdfunc
            def f(self):
                return 17

        s = G('''
            start := 'a'/f();
        ''')
        x = self.sample_start(s)
        self.assertEqual(x.d, 17)

    def test_GDenoted_choosein(self):
        class G(GrammaInterpreter):
            @gdfunc
            def f(self, x: Sample) -> str:
                return x.s

        s = G('''
            start := choose x~'somevar' in 'a'/f(x);
        ''')
        x = self.sample_start(s)
        self.assertEqual(x.d, 'somevar')

    def test_GRule_parameterized(self):
        s = GrammaInterpreter('''
            start := r('a'). ',' .r('b');
            r(x) := x . x;
        ''')
        self.assertEqual(str(self.sample_start(s)), 'aa,bb')

    def test_GRule_parameterized_random(self):
        s = GrammaInterpreter('''
            start := r('a'|'b'). ',' .r('c'|'d');
            r(x) := x{5};
        ''')
        s.random.seed(1)
        self.assertEqual(str(self.sample_start(s)), 'aaaaa,ddddd')

    def test_GRule_parameterized_nested(self):
        s = GrammaInterpreter('''
            start := r1('a'). ',' .r2('b');
            r1(x) := x . x;
            r2(x) := r1(x) . '-' . r1(r1(x));
        ''')
        self.assertEqual(str(self.sample_start(s)), 'aa,bb-bbbb')

    def test_GRule_parameterized_2_args(self):
        s = GrammaInterpreter('''
            start := r('a'|'b', 'c'|'d');
            r(x,y) := x.y;
        ''')
        s.random.seed(1)
        self.assertEqual(str(self.sample_start(s)), 'ad')

    def test_GRep_1_arg(self):
        s = GrammaInterpreter('''
            start := 'a'{3};
        ''')
        self.assertEqual(str(self.sample_start(s)), 'aaa')

    def test_GRep_hi(self):
        s = GrammaInterpreter('''
            start := 'a'{1,};
        ''')
        import gramma.samplers.interpreter
        saved_rep_high = gramma.samplers.interpreter.REP_HIGH
        gramma.samplers.interpreter.REP_HIGH = 10
        s.random.seed(1)
        self.assertEqual(str(self.sample_start(s)), 'aaa')
        gramma.samplers.interpreter.REP_HIGH = saved_rep_high

    def test_GRep_2_args(self):
        s = GrammaInterpreter('''
            start := 'a'{3,4};
        ''')
        s.random.seed(1)
        self.assertEqual(','.join(str(self.sample_start(s)) for i in range(10)),
                         'aaa,aaaa,aaaa,aaa,aaaa,aaaa,aaaa,aaa,aaa,aaa')

    def test_GRep_dists(self):
        s = GrammaInterpreter('''
               start := 'a'{geom(3)};
           ''')
        s.random.seed(1)
        self.assertEqual(','.join(str(self.sample_start(s)) for i in range(10)),
                         ',aaaa,aa,aa,,aaaaaaaa,aa,a,,a')

        s = GrammaInterpreter('''
               start := 'a'{norm(5,3)};
           ''')
        s.random.seed(1)
        self.assertEqual(','.join(str(self.sample_start(s)) for i in range(10)),
                         'aaaaaaaaaaa,aa,aaaaaaa,aaaaaa,aaaa,aaaa,aaaaaaaaaaa,aaaa,aaaaaa,aaaa')

        s = GrammaInterpreter('''
               start := 'a'{binom(5,.7)};
           ''')
        s.random.seed(1)
        self.assertEqual(','.join(str(self.sample_start(s)) for i in range(10)),
                         'aaaa,aaa,aaa,aaaa,aaaaa,aa,aaaa,aaaa,aaaa,aaaa')

        s = GrammaInterpreter('''
               start := 'a'{choice(1,2,3)};
           ''')
        s.random.seed(1)
        self.assertEqual(','.join(str(self.sample_start(s)) for i in range(10)),
                         'a,aaa,aa,aa,a,aaa,aa,aa,a,a')

    def test_GRep_gcode(self):
        class G(GrammaInterpreter):
            x: int

        s = G('''
               start := 'a'{`x-1`, `x+1`};
           ''')
        s.x = 3

        s.random.seed(1)
        self.assertEqual(','.join(str(self.sample_start(s)) for i in range(10)),
                         'aa,aaaa,aaaa,aa,aaa,aaaa,aaa,aa,aa,aa')

    def test_GRange(self):
        s = GrammaInterpreter('''
               start := ['a'..'z'];
           ''')
        s.random.seed(1)
        self.assertEqual(''.join(str(self.sample_start(s)) for i in range(10)), 'gxscoynedi')

    def test_GRange_multi(self):
        s = GrammaInterpreter('''
               start := ['a'..'z', 'A'..'Z', '0'..'9'];
           ''')
        s.random.seed(1)
        self.assertEqual(''.join(str(self.sample_start(s)) for i in range(10)), 'o3TgI5Glju')


class TestInterpreter(BaseTestInterpreter):
    def setUp(self) -> None:
        self.sample_start = GrammaInterpreter.sample_start


class TestCoroInterpreter(BaseTestInterpreter):
    def setUp(self) -> None:
        self.sample_start = GrammaInterpreter.coro_sample_start


del BaseTestInterpreter


class TestGrammars(unittest.TestCase):
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


class TestCoroGrammars(unittest.TestCase):
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

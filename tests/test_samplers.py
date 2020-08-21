#!/usr/bin/env python3

import unittest

from gramma.samplers import GrammaInterpreter, gfunc


class Arithmetic(GrammaInterpreter):
    G = '''
        start := expr;
        expr := add;
        add := mul . ('+'.mul){,3};
        mul := atom . ('*'.atom){,3};
        atom :=   'x' 
                | randint()
                | `expr_rec` '(' . expr . ')';
    '''

    def __init__(self):
        super().__init__(self.G)
        self.expr_rec = .1

    @gfunc
    def randint(self):
        return self.create_sample(str(self.random.integers(0, 100000)))


class TestInterpreter(unittest.TestCase):

    def test_tok(self):
        s = GrammaInterpreter('''
            start := 'abc';
        ''')
        self.assertEqual(str(s.sample()), 'abc')

    def test_cat(self):
        s = GrammaInterpreter('''
            start := 'a' . 'b' . 'c';
        ''')
        self.assertEqual(str(s.sample()), 'abc')

    def test_alt(self):
        s = GrammaInterpreter('''
            start := 'a' | 'b' | 'c';
        ''')
        s.random.seed(1)
        self.assertEqual(''.join(str(s.sample()) for i in range(10)), 'acbbacbbaa')

    def test_alt_numerical_weights(self):
        s = GrammaInterpreter('''
            start := 1.2 'a' | 2 'b' | 'c';
        ''')
        s.random.seed(1)
        self.assertEqual(''.join(str(s.sample()) for i in range(10)), 'abbbacbbaa')

    def test_alt_gcode_weights(self):
        s = GrammaInterpreter('''
            start := 'a' | `False` 'b' | 'c';
        ''')
        s.random.seed(1)
        self.assertEqual(''.join(str(s.sample()) for i in range(10)), 'acccacaaaa')

    def test_rules_simple(self):
        s = GrammaInterpreter('''
            start := r1 . r2 . r3;
            r1 := 'd';
            r2 := 'e';
            r3 := 'f';
        ''')
        self.assertEqual(str(s.sample()), 'def')

    def test_rules_parameterized(self):
        s = GrammaInterpreter('''
            start := r('a'). ',' .r('b');
            r(x) := x . x;
        ''')
        self.assertEqual(str(s.sample()), 'aa,bb')

    def test_rep1(self):
        s = GrammaInterpreter('''
            start := 'a'{3};
        ''')
        self.assertEqual(str(s.sample()), 'aaa')

    def test_rep2(self):
        s = GrammaInterpreter('''
            start := 'a'{3,4};
        ''')
        s.random.seed(1)
        self.assertEqual(','.join(str(s.sample()) for i in range(10)), 'aaa,aaaa,aaaa,aaa,aaaa,aaaa,aaaa,aaa,aaa,aaa')

    def test_rep_geom(self):
        s = GrammaInterpreter('''
               start := 'a'{geom(3)};
           ''')
        s.random.seed(1)
        self.assertEqual(','.join(str(s.sample()) for i in range(10)), 'a,aaaaa,aaa,aaa,a,aaaaaaaaa,aaa,aa,a,aa')

    def test_rep_gcode(self):
        class G(GrammaInterpreter):
            pass

        s = G('''
               start := 'a'{`x-1`, `x+1`};
           ''')
        s.x = 3

        s.random.seed(1)
        self.assertEqual(','.join(str(s.sample()) for i in range(10)), 'aa,aaaa,aaaa,aa,aaa,aaaa,aaa,aa,aa,aa')

    def test_basic_operators(self):
        s = GrammaInterpreter('''
            start := r1 . ','. r2{2,3} . ',' . r3;
            r1 :=  'a' | 'b';
            r2 :=  'c' | 'd';
            r3 :=  'e' | 'f';
        ''')
        s.random.seed(1)
        self.assertEqual(str(s.sample()), 'a,cdc,e')

    def test_arithmetic_grammar(self):
        s = Arithmetic()
        s.random.seed(1)
        self.assertEqual(str(s.sample()), 'x*x*17781')
        self.assertEqual(str(s.sample()), 'x+15135')
        self.assertEqual(str(s.sample()), '2810*35917*x')
        self.assertEqual(str(s.sample()), '99487*x*89714')
        self.assertEqual(str(s.sample()), 'x*94935+x*x*x')
        self.assertEqual(str(s.sample()), '77305*x*70020+x*10991*85623*'
                                          '(x*x*(x*6795*x*30102+x*x*x)*18804+x*33287*18412*x+x*x+x*x*x*x)'
                                          '+x*50515*x*x+x')


if __name__ == '__main__':
    unittest.main()

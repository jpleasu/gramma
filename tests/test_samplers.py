#!/usr/bin/env python3

import unittest

from gramma.samplers import GrammaInterpreter, gfunc


class Arithmetic(GrammaInterpreter):
    G = '''
        start := expr;
        expr := add;
        add := mul |  ('+'.mul){,3};
        mul := atom . ('*'.atom){,3};
        atom :=   'x' 
                | randint()
                | `expr_rec` '(' . expr . ')';
    '''

    def __init__(self):
        super().__init__(self.G)

    @gfunc
    def randint(self):
        return str(917)
        # return self.random


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
        self.assertEqual('acbbacbbaa', ''.join(str(s.sample()) for i in range(10)))

    def test_alt_numerical_weights(self):
        s = GrammaInterpreter('''
            start := 1.2 'a' | 2 'b' | 'c';
        ''')
        s.random.seed(1)
        self.assertEqual('abbbacbbaa', ''.join(str(s.sample()) for i in range(10)))

    def test_alt_gcode_weights(self):
        s = GrammaInterpreter('''
            start := 'a' | `False` 'b' | 'c';
        ''')
        s.random.seed(1)
        self.assertEqual('acccacaaaa', ''.join(str(s.sample()) for i in range(10)))

    def test_rule(self):
        s = GrammaInterpreter('''
            start := r1 . r2 . r3;
            r1 := 'd';
            r2 := 'e';
            r3 := 'f';
        ''')
        self.assertEqual(str(s.sample()), 'def')

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
        self.assertEqual('a,cdc,e', str(s.sample()))


if __name__ == '__main__':
    unittest.main()

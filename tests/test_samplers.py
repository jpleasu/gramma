#!/usr/bin/env python3

import unittest

from gramma.samplers import GrammaInterpreterSamplerBase, gfunc


class ArithmeticSampler(GrammaInterpreterSamplerBase):
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


class TestInterpreters(unittest.TestCase):
    def test_simple(self):
        s = ArithmeticSampler()


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3

import unittest

import filelock

from gramma.samplers import GrammaInterpreter
from gramma.samplers.cpp.randomapi import RandomAPI, DLL_PATH

try:
    from .cpp_testing_common import CppTestMixin
except ImportError:
    # noinspection PyUnresolvedReferences
    from cpp_testing_common import CppTestMixin  # type: ignore


def get_randomapi():
    """
    tests should politely wait their turn if the DLL needs building.
    tox deletes DLL_PATH
    """
    with filelock.FileLock(DLL_PATH + '.test_lock', timeout=600):
        return RandomAPI()


class TestKnownValues(unittest.TestCase):
    def test_integers(self):
        r = get_randomapi()
        r.seed(1)
        self.assertEqual(r.integers(0, 30), 4)
        r.seed(1)
        self.assertEqual(r.integers(0, 30), 4)
        self.assertEqual(r.integers(1000, 100000), 14504)

    def test_geometric(self):
        r = get_randomapi()
        r.seed(1)
        self.assertEqual(r.geometric(1 / 30), 4)
        r.seed(1)
        self.assertEqual(r.geometric(1 / 30), 4)
        self.assertEqual(r.geometric(1 / 100000), 14665)

    def test_normal(self):
        r = get_randomapi()
        r.seed(1)
        self.assertEqual(r.normal(30, 1), 29.61316823837896)
        r.seed(1)
        self.assertEqual(r.normal(30, 1), 29.61316823837896)
        self.assertEqual(r.normal(100000, 10000), 106868.23639179325)

    def test_binomial(self):
        r = get_randomapi()
        r.seed(1)
        self.assertEqual(r.binomial(30, .1), 5)
        r.seed(1)
        self.assertEqual(r.binomial(30, .1), 5)
        self.assertEqual(r.binomial(100000, .5), 50019)


class TestCompareWithInterpreter(unittest.TestCase, CppTestMixin):
    def test_alt(self):
        glf = '''
            start:= ('a'|'b')
                   .('c'|2 'd')
                   .('g'|0 'h')
                   .(.2234234 'i'|.88383 'j')
            ;
        '''
        g = GrammaInterpreter(glf)
        g.random = get_randomapi()
        seedval = 1
        g.random.seed(seedval)
        interpreted = ''.join(str(g.sample_start()) for i in range(30))
        self.assertSampleEquals(glf, interpreted, count=30, seed=seedval)

    def test_tern(self):
        """
        the gcode portions must be valid in Python and C++
        """
        glf = '''
            start:= (`random.normal(0,1)>0`? 'a' : 'b')
                  . (`random.binomial(1,.5)`? 'c' : 'd')
                  . (`random.geometric(.01)&1`? 'e' : 'f')
            ;
        '''
        g = GrammaInterpreter(glf)
        g.random = get_randomapi()
        seedval = 1
        g.random.seed(seedval)
        interpreted = ''.join(str(g.sample_start()) for i in range(30))
        self.assertSampleEquals(glf, interpreted, count=30, seed=seedval)

    def test_rep(self):
        glf = '''
            start:=('a'|'b'|'1'){30}
                  .('c'|'d'|'2'){10,30}
                  .('e'|'f'|'3'){,30}
                  .('g'|'h'|'4'){geom(15)}
                  .('i'|'j'|'5'){norm(15,5)}
                  .('k'|'l'|'6'){binom(15,.5)}
                  .('m'|'n'|'7'){choice(2,3,4,5)}
            ;
        '''
        g = GrammaInterpreter(glf)
        g.random = get_randomapi()
        seedval = 3
        g.random.seed(seedval)
        interpreted = str(g.sample_start())
        self.assertSampleEquals(glf, interpreted, count=1, seed=seedval)

    def test_range(self):
        glf = '''
            start:=['a'..'z']
                  .','
                  .['a'..'z', 'A'..'Z']
                  .','
                  .['0'..'9','a'..'z', 'A'..'Z']
                  .','
                  .['a','e','i','o','u']
            ;
        '''
        g = GrammaInterpreter(glf)
        g.random = get_randomapi()
        seedval = 7
        g.random.seed(seedval)
        interpreted = ''.join(str(g.sample_start()) for i in range(30))
        self.assertSampleEquals(glf, interpreted, count=30, seed=seedval)


if __name__ == '__main__':
    unittest.main()

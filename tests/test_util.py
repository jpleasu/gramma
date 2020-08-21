#!/usr/bin/env python3

import unittest

from gramma.util import SetStack, DictStack


class TestSetStack(unittest.TestCase):

    def test_basic(self):
        s = SetStack()
        s.add('a')
        s.push()
        s.add('b')
        self.assertTrue('a' in s)
        self.assertTrue('b' in s)
        self.assertFalse('c' in s)
        s.push()
        self.assertTrue('a' in s)
        self.assertTrue('b' in s)
        self.assertFalse('c' in s)
        s.pop()
        self.assertTrue('a' in s)
        self.assertTrue('b' in s)
        self.assertFalse('c' in s)
        s.pop()
        self.assertTrue('a' in s)
        self.assertFalse('b' in s)
        self.assertFalse('c' in s)
        s.pop()
        self.assertFalse('a' in s)
        self.assertFalse('b' in s)
        self.assertFalse('c' in s)

    def test_contexts(self):
        s = SetStack()
        with s.context(set(['a'])):
            self.assertTrue('a' in s)
            self.assertFalse('b' in s)
            self.assertFalse('c' in s)
            with s.context(set(['b'])):
                self.assertTrue('a' in s)
                self.assertTrue('b' in s)
                self.assertFalse('c' in s)
                with s.context():
                    self.assertTrue('a' in s)
                    self.assertTrue('b' in s)
                    self.assertFalse('c' in s)
                self.assertTrue('a' in s)
                self.assertTrue('b' in s)
                self.assertFalse('c' in s)
            self.assertTrue('a' in s)
            self.assertFalse('b' in s)
            self.assertFalse('c' in s)
        self.assertFalse('a' in s)
        self.assertFalse('b' in s)
        self.assertFalse('c' in s)


class TestDictStack(unittest.TestCase):

    def test_basic(self):
        d = DictStack()
        d['a'] = 1
        d.push()
        d['b'] = 2
        self.assertEqual(d['a'], 1)
        self.assertEqual(d['b'], 2)
        self.assertFalse('c' in d)
        d.push()
        self.assertEqual(d['a'], 1)
        self.assertEqual(d['b'], 2)
        self.assertFalse('c' in d)
        d.pop()
        self.assertEqual(d['a'], 1)
        self.assertEqual(d['b'], 2)
        self.assertFalse('c' in d)
        d.pop()
        self.assertEqual(d['a'], 1)
        self.assertFalse('b' in d)
        self.assertFalse('c' in d)
        d.pop()
        self.assertFalse('a' in d)
        self.assertFalse('b' in d)
        self.assertFalse('c' in d)

    def test_contexts(self):
        d = DictStack()
        with d.context(dict(a=1)):
            self.assertEqual(d['a'], 1)
            self.assertFalse('b' in d)
            self.assertFalse('c' in d)
            with d.context(dict(b=2)):
                self.assertEqual(d['a'], 1)
                self.assertEqual(d['b'], 2)
                self.assertFalse('c' in d)
                with d.context():
                    self.assertEqual(d['a'], 1)
                    self.assertEqual(d['b'], 2)
                    self.assertFalse('c' in d)
                self.assertEqual(d['a'], 1)
                self.assertEqual(d['b'], 2)
                self.assertFalse('c' in d)
            self.assertEqual(d['a'], 1)
            self.assertFalse('b' in d)
            self.assertFalse('c' in d)
        self.assertFalse('a' in d)
        self.assertFalse('b' in d)
        self.assertFalse('c' in d)


if __name__ == '__main__':
    unittest.main()

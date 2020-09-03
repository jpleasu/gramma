#!/usr/bin/env python3

import unittest
from io import StringIO

from gramma.util import SetStack, DictStack
from gramma.util.emitters import Emitter, EmitterError

import tempfile

import os

import pytest


class TestEmitter(unittest.TestCase):

    def test_write_to_string(self):
        e = Emitter()
        tmpfilename = tempfile.mktemp()
        try:
            e.write_to(tmpfilename)
            e.emit('abcd')
            e.close()

            with open(tmpfilename, 'r') as infile:
                result = infile.read()

            self.assertEqual(result, 'abcd\n')
        finally:
            if os.path.exists(tmpfilename):
                os.unlink(tmpfilename)

    def test_write_context(self):
        sio = StringIO()
        e = Emitter()
        with e.write_to(sio):
            e.emit('abcd')
            result = sio.getvalue()
        self.assertIsNone(e.out)
        self.assertEqual(result, 'abcd\n')

    def test_echo(self):
        sio = StringIO()
        eio = StringIO()

        e = Emitter(out=sio, echo=eio)
        e.emit('abcd')
        self.assertEqual(sio.getvalue(), 'abcd\n')
        self.assertEqual(eio.getvalue(), 'abcd\n')

        sio.close()
        eio.close()

    def test_except_if_no_out(self):
        with pytest.raises(EmitterError):
            e = Emitter()
            e.emit('anything')

    def test_afters(self):
        sio = StringIO()
        e = Emitter(out=sio)
        e.emit('1')
        e.emit('4', after='x')
        e.emit('2')
        e.emit('3', tag='x')
        self.assertEqual(sio.getvalue(), '1\n2\n3\n4\n')

        sio.close()


class TestSetStack(unittest.TestCase):

    def test_basic(self):
        s = SetStack()
        s.add('a')
        s.push()
        s.add('b')
        self.assertIn('a', s)
        self.assertIn('b', s)
        self.assertNotIn('c', s)
        s.push()
        self.assertIn('a', s)
        self.assertIn('b', s)
        self.assertNotIn('c', s)
        s.pop()
        self.assertIn('a', s)
        self.assertIn('b', s)
        self.assertNotIn('c', s)
        s.pop()
        self.assertIn('a', s)
        self.assertNotIn('b', s)
        self.assertNotIn('c', s)
        s.pop()
        self.assertNotIn('a', s)
        self.assertNotIn('b', s)
        self.assertNotIn('c', s)

    def test_contexts(self):
        s = SetStack()
        with s.context({'a'}):
            self.assertIn('a', s)
            self.assertNotIn('b', s)
            self.assertNotIn('c', s)
            with s.context(set(['b'])):
                self.assertIn('a', s)
                self.assertIn('b', s)
                self.assertNotIn('c', s)
                with s.context():
                    self.assertIn('a', s)
                    self.assertIn('b', s)
                    self.assertNotIn('c', s)
                self.assertIn('a', s)
                self.assertIn('b', s)
                self.assertNotIn('c', s)
            self.assertIn('a', s)
            self.assertNotIn('b', s)
            self.assertNotIn('c', s)
        self.assertNotIn('a', s)
        self.assertNotIn('b', s)
        self.assertNotIn('c', s)


class TestDictStack(unittest.TestCase):

    def test_basic(self):
        d = DictStack()
        d['a'] = 1
        d.push()
        d['b'] = 2
        self.assertIn('a', d)
        self.assertEqual(d['a'], 1)
        self.assertIn('b', d)
        self.assertEqual(d['b'], 2)
        self.assertNotIn('c', d)
        d.push()
        self.assertIn('a', d)
        self.assertEqual(d['a'], 1)
        self.assertIn('b', d)
        self.assertEqual(d['b'], 2)
        self.assertNotIn('c', d)
        d.pop()
        self.assertIn('a', d)
        self.assertEqual(d['a'], 1)
        self.assertIn('b', d)
        self.assertEqual(d['b'], 2)
        self.assertNotIn('c', d)
        d.pop()
        self.assertIn('a', d)
        self.assertEqual(d['a'], 1)
        self.assertNotIn('b', d)
        self.assertNotIn('c', d)
        d.pop()
        self.assertNotIn('a', d)
        self.assertNotIn('b', d)
        self.assertNotIn('c', d)

    def test_get_missing(self):
        d = DictStack()
        d['a'] = 1
        d.push()
        d['b'] = 2
        self.assertIsNone(d.get('c'))
        d.push()
        self.assertIsNone(d.get('c'))
        d.pop()
        self.assertIsNone(d.get('c'))
        d.pop()
        self.assertIsNone(d.get('c'))

    def test_contexts(self):
        d = DictStack()
        with d.context(dict(a=1)):
            self.assertEqual(d['a'], 1)
            self.assertNotIn('b', d)
            self.assertNotIn('c', d)
            with d.context(dict(b=2)):
                self.assertEqual(d['a'], 1)
                self.assertEqual(d['b'], 2)
                self.assertNotIn('c', d)
                with d.context():
                    self.assertEqual(d['a'], 1)
                    self.assertEqual(d['b'], 2)
                    self.assertNotIn('c', d)
                self.assertEqual(d['a'], 1)
                self.assertEqual(d['b'], 2)
                self.assertNotIn('c', d)
            self.assertEqual(d['a'], 1)
            self.assertNotIn('b', d)
            self.assertNotIn('c', d)
        self.assertNotIn('a', d)
        self.assertNotIn('b', d)
        self.assertNotIn('c', d)

    def test_str(self):
        d = DictStack()
        d['a'] = 1
        self.assertEqual(str(d), "{'a': 1}")
        d.push()
        self.assertEqual(str(d), "{'a': 1}")
        d['b'] = 2
        self.assertEqual(str(d), "{'a': 1},{'b': 2}")
        d.push()
        self.assertEqual(str(d), "{'a': 1},{'b': 2}")
        d.pop()
        self.assertEqual(str(d), "{'a': 1},{'b': 2}")
        d.pop()
        self.assertEqual(str(d), "{'a': 1}")
        d.pop()
        self.assertEqual(str(d), "{}")


if __name__ == '__main__':
    unittest.main()

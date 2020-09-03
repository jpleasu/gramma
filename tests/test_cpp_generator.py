#!/usr/bin/env python3

import unittest
from io import StringIO
from typing import cast, Optional
import os
import sys
import shutil
import subprocess

import tempfile

from gramma.parser import GrammaGrammar, GFuncRef, GRuleRef, GChooseIn, GVarRef, GAlt, GRep, RepDist, GRange, \
    GDenoted, GTok, GCode, GDFuncRef

from gramma.samplers.cpp.glf2cpp import CppEmitter, INCLUDE_DIR

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), '..', 'examples')

_CXX: Optional[str] = os.environ.get('CXX')
if _CXX is None:
    _CXX = shutil.which('clang++') or shutil.which('gcc')
if _CXX is None:
    raise SystemError('no compiler found')
CXX: str = _CXX


class TestInvokes(unittest.TestCase):
    """
    the C++ generator treats
    """

    def assertSampleEquals(self, glf: str, expected: str, count: Optional[int] = 10, seed: int = 1) -> None:
        g = GrammaGrammar(glf)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', encoding='utf8') as tmpsourcefile:
            e = CppEmitter(g, 'test_grammar', out=tmpsourcefile, echo=sys.stdout)
            e.write_monolithic_main(count=count, seed=seed, close=False)

            tmpexecutable = tempfile.NamedTemporaryFile(dir=os.getcwd(), delete=False)

            try:
                result = subprocess.call(
                    [CXX, '-I', INCLUDE_DIR, '-o', tmpexecutable.name, tmpsourcefile.name])
                self.assertEqual(result, 0, 'failed to build')
                tmpexecutable.close()

                out = subprocess.check_output([tmpexecutable.name], encoding='utf8')
                self.assertEqual(out, expected)
            finally:
                if os.path.exists(tmpexecutable.name):
                    os.unlink(tmpexecutable.name)

    def test_GCat(self):
        self.assertSampleEquals('''
            start := 'a' . 'b';
        ''', 'ab' * 10)

    def test_GAlt(self):
        self.assertSampleEquals('''
            start := 'a' | 'b';
        ''', 'aaaaabaabb')

    def test_GTern(self):
        self.assertSampleEquals('''
            start := `random.uniform()<.7` ? 'a' : 'b';
        ''', 'aaaaabaaaa')

    def test_GChooseIn(self):
        self.assertSampleEquals('''
            start := choose x ~ ('a'|'b') in x.x.x.' ';
        ''', 'aaa aaa aaa aaa aaa bbb aaa aaa bbb bbb ')

    def test_GRange(self):
        self.assertSampleEquals('''
            start := ['a'..'z'];
        ''', 'ddlajxmboq')

    def test_GRep(self):
        self.assertSampleEquals('''
            start := 'a'{1,5}.' ';
        ''', 'a a aaa a aa aaaaa aaa a aaa aaaa ')

    def test_GDenoted(self):
        self.assertSampleEquals('''
            start := show_den('a'/1 | 'b'/2 | 'c'/3). ' ';
        ''', 'a<1> a<1> b<2> a<1> b<2> c<3> b<2> a<1> b<2> b<2> ')

    def test_GRule(self):
        self.assertSampleEquals('''
            start := r('a'|'b', 'c'|'d'). ' ';
            r(x,y) := x.x.y.y;
        ''', 'aacc aacc aadd aacc bbdd aadd bbcc aacc aadd aacc ')


if __name__ == '__main__':
    unittest.main()

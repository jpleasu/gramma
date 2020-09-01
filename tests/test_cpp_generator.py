#!/usr/bin/env python3

import unittest
from io import StringIO
from typing import cast
import os
import sys
import shutil
import subprocess

import tempfile

from gramma.parser import GrammaGrammar, GFuncRef, GRuleRef, GChooseIn, GVarRef, GAlt, GRep, RepDist, GRange, \
    GDenoted, GTok, GCode, GDFuncRef

from gramma.samplers.generators.cpp import CppEmitter

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), '..', 'examples')
INCLUDE_DIR = os.path.join(os.path.dirname(__file__), '..', 'include')

cxx = shutil.which('clang++') or shutil.which('gcc')


def trybuild(sourcefile):
    try:
        with tempfile.NamedTemporaryFile() as tmpfile:
            if subprocess.call([cxx, '-I', INCLUDE_DIR, '-o', tmpfile.name, sourcefile]) == 0:
                return True
    except:
        pass
    return False


class TestInvokes(unittest.TestCase):
    """
    the C++ generator treats
    """

    def assertSampleEquals(self, glf, sample):
        g = GrammaGrammar(glf)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', encoding='utf8') as tmpsourcefile:
            e = CppEmitter(tmpsourcefile, g, 'test_grammar', echo=sys.stderr)
            e.emit('''\
                #include <string>
                #include <iostream>
            ''')

            e.emit_base()

            e.emit('''\
                // a sampler
                class test_grammar: public test_grammar_base<test_grammar, std::string, int> {
                    public:
                    std::string cat(const std::string &a, const std::string &b) {
                        return a+b;
                    }
                };
            
                // and main
                int main() {
                    auto s=test_grammar();
                    s.random.set_seed(1);
                    std::cout << s.start() << "\\n";
                    return 0;
                }
            ''')

            tmpexecutable = tempfile.NamedTemporaryFile(dir=os.getcwd(), delete=False)
            try:
                result = subprocess.call([cxx, '-I', INCLUDE_DIR, '-o', tmpexecutable.name, tmpsourcefile.name])
                self.assertEqual(result, 0, 'failed to build')
                tmpexecutable.close()

                out = subprocess.check_output([tmpexecutable.name], encoding='utf8')
                self.assertEqual(out, sample)
            finally:
                if os.path.exists(tmpexecutable.name):
                    os.unlink(tmpexecutable.name)

    def test_GCat(self):
        self.assertSampleEquals('''
            start := 'a' . 'b';
        ''', 'ab\n')

    def xtest_variety(self):
        with open(os.path.join(EXAMPLE_DIR, 'variety', 'variety.glf')) as infile:
            g = GrammaGrammar(infile.read())

        sio = StringIO()
        e = CppEmitter(sio, g, 'variety')
        e.emit_base()
        print(sio.getvalue())


if __name__ == '__main__':
    unittest.main()

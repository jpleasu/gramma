#!/usr/bin/env python3

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from io import StringIO
from typing import cast, Optional

from gramma.parser import GrammaGrammar, GExpr, GrammaParseError
from gramma.samplers.cpp.glf2cpp import CppEmitter, INCLUDE_DIR, CXXFLAGS, shell_join, get_compiler, encode_as_cpp, \
    CppEmitterError

from gramma.samplers.cpp.glf2cpp import main as glf2cpp_main

EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), '..', 'examples')

CXX: Optional[str] = get_compiler()


def returns_none(*args):
    return None


class TestUtils(unittest.TestCase):
    def test_CXX(self):
        self.assertIsNotNone(CXX, 'no compiler found!')

    def test_shell_args(self):
        self.assertEqual(shell_join(['a', 'b', 'c']), 'a b c')
        self.assertEqual(shell_join(['a""a', "b''b", 'c c']), r'''"a\"\"a" "b''b" "c c"''')

    def test_encode_as_cpp(self):
        self.assertEqual(encode_as_cpp('', '"'), '""')
        self.assertEqual(encode_as_cpp(b'abc', '"'), r'"abc"')
        self.assertEqual(encode_as_cpp(b'a\x00c', '"'), r'"a\0c"')
        self.assertEqual(encode_as_cpp(b'a\xffc', '"'), r'"a\xffc"')
        self.assertEqual(encode_as_cpp(ord('a'), '"'), r'"a"')
        self.assertEqual(encode_as_cpp('"', '"'), r'"\""')
        self.assertEqual(encode_as_cpp('\\', '"'), r'"\\"')
        self.assertEqual(encode_as_cpp('"\\\n\r\t', '"'), r'"\"\\\n\r\t"')

    def test_cpp_emitter(self):
        sio = StringIO()
        eio = StringIO()

        grammar = GrammaGrammar("start := 'a';")
        emitter = CppEmitter(grammar, 'dummy_sampler', echo=eio)
        emitter.write_monolithic_main(out=sio)
        self.assertEqual(eio.getvalue()[:8], '#include')
        self.assertIsNone(emitter.out)
        with self.assertRaises(ValueError):
            sio.getvalue()
        eio.close()


class GFake(GExpr):
    pass


class TestHandlers(unittest.TestCase):
    def test_method(self):
        with self.assertRaises(CppEmitterError):
            grammar = GrammaGrammar("start := 'a';")
            emitter = CppEmitter(grammar, 'dummy_sampler')
            emitter.emit_method(GFake(), 'definition')

    def test_gdfunc_args(self):
        with self.assertRaises(CppEmitterError):
            grammar = GrammaGrammar("start := 'a';")
            emitter = CppEmitter(grammar, 'dummy_sampler')
            emitter.as_gdfunc_arg(GFake())

    def test_invoke_rep_dist(self):
        with self.assertRaises(GrammaParseError):
            grammar = GrammaGrammar("start := 'a'{fake(1,2,3)};")
            sio = StringIO()
            emitter = CppEmitter(grammar, 'dummy_sampler')
            with emitter.write_to(sio):
                emitter.write_monolithic_main()


class TestNodes(unittest.TestCase):
    def assertSampleEquals(self, glf: str, expected: str, count: Optional[int] = 10, seed: int = 1) -> None:
        g = GrammaGrammar(glf)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', encoding='utf8') as tmpsourcefile:
            e = CppEmitter(g, 'test_grammar', out=tmpsourcefile, echo=sys.stdout)

            tmpexecutable = tempfile.NamedTemporaryFile(dir=os.getcwd(), delete=False)

            cmd_args = [cast(str, CXX)] + CXXFLAGS + ['-I', INCLUDE_DIR, '-o', tmpexecutable.name, tmpsourcefile.name]
            e.emit('// ' + shell_join(cmd_args))
            e.write_monolithic_main(count=count, seed=seed, close=False)

            try:
                result = subprocess.call(cmd_args)
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
        ''', 'ab', count=1)

    def test_GAlt(self):
        self.assertSampleEquals('''
            start := 'a' | 'b';
        ''', 'aaaaabaabb')

    def test_GAlt_dynamic(self):
        self.assertSampleEquals('''
            start := 'a' | `false` 'b' | 'c';
        ''', 'aaaaacaacc')

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

    def test_GRange_multibyte(self):
        self.assertSampleEquals('''
            start := ['😇','😈'];
        ''', '😇😇😇😇😇😈😇😇😈😈')

    def test_GRep(self):
        self.assertSampleEquals('''
            start := 'a'{1,5}.' ';
        ''', 'a a aaa a aa aaaaa aaa a aaa aaaa ')

    def test_GRep_blank_lo(self):
        self.assertSampleEquals('''
            start := 'a'{,2}.' ';
        ''', '  a  a aa a  a a ')

    def test_GRep_fixed(self):
        self.assertSampleEquals('''
            start := 'a'{2}.' ';
        ''', 'aa aa aa aa aa aa aa aa aa aa ')

    def test_GRep_code(self):
        self.assertSampleEquals('''
            start := 'a'{`1`,`1+1`}.' ';
        ''', 'a a a a a aa a a aa aa ')

    def test_GRep_dists(self):
        self.assertSampleEquals('''
            start := 'a'{geom(3)}.' ';
        ''', '  a   aaaaaaa a  a aa ')
        self.assertSampleEquals('''
            start := 'a'{norm(5,2)}.' ';
        ''', 'aaaa aaaaaa aaa aaaaaaaaa aaaaa aaaa aa aaaaaaa aa aaaaaaa ')
        self.assertSampleEquals('''
            start := 'a'{binom(5,.7)}.' ';
        ''', 'a aaaaa aaa aaa aaaa aa aa aa aaa aaaa ')
        self.assertSampleEquals('''
            start := 'a'{choice(1,2,3)}.' ';
        ''', 'a a aa a aa aaa aa a aa aa ')

    def test_GDenoted(self):
        self.assertSampleEquals('''
            start := show_den('a'/1 | 'b'/2 | 'c'/3). ' ';
        ''', 'a<1> a<1> b<2> a<1> b<2> c<3> b<2> a<1> b<2> b<2> ')

    def test_GFunc(self):
        self.assertSampleEquals('''
            start := choose x~'c' in x2('a').x2(x2('b')).x2(x).x2(`"d"`).x2(r);
            r := 'e';
        ''', 'aabbbbccddee', count=1)

    def test_GFunc_stubs(self):
        self.assertSampleEquals('''
            start := f().g();
        ''', '(f stub)(g stub)', count=1)

    def test_GDFunc(self):
        self.assertSampleEquals('''
            start := show_den('a'/gdf(1));
        ''', 'a<(gdf stub)>', count=1)

    def test_GDFunc_asarg(self):
        self.assertSampleEquals('''
            start := show_den('a'/gdf(gdf(1)));
        ''', 'a<(gdf stub)>', count=1)

    def test_GDFunc_var(self):
        """sample decays to string"""
        self.assertSampleEquals('''
            start := choose x~('b'/2) in show_den('a'/x);
        ''', 'a<b>', count=1)

    def test_GDFunc_var2(self):
        """default gdfunc argument type of variant will also decay to string"""
        self.assertSampleEquals('''
            start := choose x~'a' in show_den('a'/gdf(x));
        ''', 'a<(gdf stub)>', count=1)

    def test_GDFunc_var3(self):
        """get at the sample's denotation with get_den, a function with sample_t arg"""
        self.assertSampleEquals('''
            start := choose x~('b'/2) in show_den('a'/get_den(x));
        ''', 'a<2>', count=1)

    def test_GDFunc_code(self):
        self.assertSampleEquals('''
            start := show_den('a'/`1+1`);
        ''', 'a<2>', count=1)

    def test_GDFunc_code2(self):
        self.assertSampleEquals('''
            start := show_den('a'/gdf(`1+1`));
        ''', 'a<(gdf stub)>', count=1)

    def test_GRule(self):
        self.assertSampleEquals('''
            start := r('a'|'b', 'c'|'d'). ' ';
            r(x,y) := x.x.y.y;
        ''', 'aacc aacc aadd aacc bbdd aadd bbcc aacc aadd aacc ')


class TestMain(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.TemporaryDirectory()
        os.chdir(self.test_dir.name)

    def tearDown(self) -> None:
        self.test_dir.cleanup()

    def assertFileExists(self, path: str) -> None:
        self.assertTrue(os.path.exists(path), f'file missing: {path}')

    def assertFileDoesNotExist(self, path: str) -> None:
        self.assertFalse(os.path.exists(path), f'file present: {path}')

    def test_build_dummy(self):
        glf_path = os.path.join(os.path.dirname(__file__), 'dummy.glf')
        self.assertFileExists(glf_path)
        generated_files = [
            'dummy_sampler.cpp',
            'dummy_sampler_def.inc',
            'dummy_sampler_decl.inc',
            'dummy_sampler',  # + '.exe'
        ]
        for path in generated_files:
            self.assertFileDoesNotExist(path)
        try:
            glf2cpp_main([glf_path, '-m', '-b'])
            for path in generated_files:
                self.assertFileExists(path)
        finally:
            for path in generated_files:
                if os.path.exists(path):
                    os.unlink(path)

    def test_build_dont_build_without_compiler(self):
        glf_path = os.path.join(os.path.dirname(__file__), 'dummy.glf')
        self.assertFileExists(glf_path)
        generated_files = [
            'dummy_sampler.cpp',
            'dummy_sampler_def.inc',
            'dummy_sampler_decl.inc',
            'dummy_sampler',  # + '.exe'
        ]
        for path in generated_files:
            self.assertFileDoesNotExist(path)
        try:
            import gramma.samplers.cpp.glf2cpp
            saved_get_compiler = gramma.samplers.cpp.glf2cpp.get_compiler
            gramma.samplers.cpp.glf2cpp.get_compiler = returns_none
            try:
                glf2cpp_main([glf_path, '-m', '-b'])
                for path in generated_files[:3]:
                    self.assertFileExists(path)
                self.assertFileDoesNotExist(generated_files[3])
            finally:
                gramma.samplers.cpp.glf2cpp.get_compiler = saved_get_compiler
        finally:
            for path in generated_files:
                if os.path.exists(path):
                    os.unlink(path)

    def test_build_dummy_module(self):
        glf_path = os.path.join(os.path.dirname(__file__), 'dummy.glf')
        self.assertFileExists(glf_path)
        generated_files = [
            'dummy_sampler.cpp',
            'dummy_sampler_def.inc',
            'dummy_sampler_decl.inc',
            'dummy_sampler.so',  # '.so'->'.dll'
        ]
        for path in generated_files:
            self.assertFileDoesNotExist(path)
        try:
            glf2cpp_main([glf_path, '-b'])
            for path in generated_files:
                self.assertFileExists(path)
        finally:
            for path in generated_files:
                if os.path.exists(path):
                    os.unlink(path)

    def test_build_dummy_named_sampler(self):
        glf_path = os.path.join(os.path.dirname(__file__), 'dummy.glf')
        sampler_name = 'idiot_sampler'
        self.assertFileExists(glf_path)
        generated_files = [
            'idiot_sampler.cpp',
            'idiot_sampler_def.inc',
            'idiot_sampler_decl.inc',
            'idiota',  # + '.exe'
        ]
        for path in generated_files:
            self.assertFileDoesNotExist(path)
        try:
            glf2cpp_main([glf_path, '-m', '-b', '-s', sampler_name, '-b', 'idiota'])
            for path in generated_files:
                self.assertFileExists(path)
        finally:
            for path in generated_files:
                if os.path.exists(path):
                    os.unlink(path)

    def test_build_dummy_forcing(self):
        glf_path = os.path.join(os.path.dirname(__file__), 'dummy.glf')
        self.assertFileExists(glf_path)
        generated_files = [
            'dummy_sampler.cpp',
            'dummy_sampler_def.inc',
            'dummy_sampler_decl.inc',
            'dummy_sampler',
        ]
        for path in generated_files:
            self.assertFileDoesNotExist(path)
        try:
            glf2cpp_main([glf_path, '-m', '-b'])
            for path in generated_files:
                self.assertFileExists(path)
            mtimes = [os.path.getmtime(path) for path in generated_files]

            glf2cpp_main([glf_path, '-m', '-b'])
            for mtime, path in zip(mtimes, generated_files[:3]):
                self.assertEqual(mtime, os.path.getmtime(path))
            self.assertLess(mtimes[3], os.path.getmtime(generated_files[3]))
            mtimes = [os.path.getmtime(path) for path in generated_files]

            glf2cpp_main([glf_path, '-m', '-b', '-f'])
            self.assertEqual(mtimes[0], os.path.getmtime(generated_files[0]))
            for mtime, path in zip(mtimes, generated_files[1:]):
                self.assertLess(mtime, os.path.getmtime(path))
            mtimes = [os.path.getmtime(path) for path in generated_files]

            glf2cpp_main([glf_path, '-m', '-b', '-ff'])
            for mtime, path in zip(mtimes, generated_files):
                self.assertLess(mtime, os.path.getmtime(path))
            mtimes = [os.path.getmtime(path) for path in generated_files]

        finally:
            for path in generated_files:
                if os.path.exists(path):
                    os.unlink(path)


if __name__ == '__main__':
    unittest.main()

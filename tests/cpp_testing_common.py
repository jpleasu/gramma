import os
import subprocess
import tempfile
from typing import cast, Optional, Any

from gramma import GrammaGrammar
from gramma.samplers.cpp.glf2cpp import CppEmitter, CXXFLAGS, INCLUDE_DIR, shell_join, get_compiler

CXX: Optional[str] = get_compiler()

import sys


# noinspection PyPep8Naming
class CppTestMixin:
    def assertEqual(self, a: Any, b: Any, msg: Optional[str] = None) -> None:
        ...

    def assertSampleEquals(self, glf: str, expected: str, count: Optional[int] = 10, seed: int = 1,
                           enforce_ltr: bool = True) -> None:
        g = GrammaGrammar(glf)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', encoding='utf8') as tmpsourcefile:
            e = CppEmitter(g, 'test_grammar', out=tmpsourcefile, echo=None, enforce_ltr=enforce_ltr)

            tmpexecutable = tempfile.NamedTemporaryFile(delete=False)

            try:
                cmd_args = [cast(str, CXX)] + CXXFLAGS + ['-I', INCLUDE_DIR, '-o', tmpexecutable.name,
                                                          tmpsourcefile.name]
                e.emit('// ' + shell_join(cmd_args))
                e.write_monolithic_main(count=count, seed=seed, close=False)

                result = subprocess.call(cmd_args)
                self.assertEqual(result, 0, 'failed to build')
                tmpexecutable.close()

                out = subprocess.check_output([tmpexecutable.name], encoding='utf8')
                self.assertEqual(out, expected)
            finally:
                if os.path.exists(tmpexecutable.name):
                    os.unlink(tmpexecutable.name)

#!/usr/bin/env python3
r"""
C++ sampler support


imple header needs forward def of sampler class for SamplerBase parameter
sampler header needs impl header
    to inherit
    for sample_type


TODO
- support wchar_t
    - if we're in multibyte mode, use ENCODING, else, emit character to C++ file and use L'xxx' or L'\u....'
      with wchar_t
    - see emit_method_GRange
- add gdfuncs and denotation parsing

"""
import logging
import os
import sys
from typing import Dict, cast, List, Union, Iterable, Optional, IO, Set, Callable, Tuple, Literal

from gramma.util.emitters import Emitter
from ...parser import GExpr, GTok, GFuncRef, GInternal, GVarRef, GCode, GRuleRef, RepDist, \
    GrammaParseError, GAlt, GDenoted, GCat, GTern, GChooseIn, GRange, GRep, GrammaGrammar, GDFuncRef

log = logging.getLogger('gramma.samplers.cpp')

# TODO this is used to encode "wide" chars as UTF8 in multibyte mode
ENCODING = 'utf8'

INCLUDE_DIR = os.path.join(os.path.dirname(__file__), 'include')

EmitMode = Literal['definition', 'declaration']


def encode_as_cpp(s: Union[str, bytes, int], quote: str) -> str:
    quoteord = ord(quote)
    r = quote
    b: Iterable[int]
    if isinstance(s, bytes):
        b = s
    elif isinstance(s, int):
        b = (s,)
    else:
        b = s.encode(ENCODING)
    for c in b:
        if c == 0:
            r += r'\0'
        elif c == 10:
            r += r'\n'
        elif c == 13:
            r += r'\r'
        elif c == 9:
            r += r'\t'
        elif c == quoteord:
            r += '\\' + quote
        elif c == 92:
            r += r'\\'  # two backslashes
        elif 0x20 <= c <= 0x7e:
            r += chr(c)
        else:
            r += f'\\x{c:02x}'
    return r + quote


def encode_as_cpp_str(s: Union[str, bytes]) -> str:
    return encode_as_cpp(s, quote='"')


def encode_as_cpp_char(c: str) -> str:
    return encode_as_cpp(ord(c), quote="'")


class CppEmitterError(Exception):
    pass


# noinspection PyPep8Naming
class CppEmitter(Emitter):
    grammar: GrammaGrammar
    sampler_name: str
    impl_name: str

    ident: Dict[GExpr, str]
    varids: Set[str]  # (vname)
    gfs: Set[Tuple[str, int]]  # (fname, nargs)

    def __init__(self, grammar: GrammaGrammar, sampler_name: str,
                 out: Union[None, str, IO[str]] = None, echo: Optional[IO[str]] = None):
        Emitter.__init__(self, out, echo)

        self.grammar = grammar
        self.sampler_name = sampler_name
        self.impl_name = sampler_name + '_impl'

        # walk the AST to collect info
        self.ident = {}
        self.varids = set()
        self.gfs = set()
        for ruledef in self.grammar.ruledefs.values():
            self.walk_ast(ruledef.rhs)

    def walk_ast(self, ge: GExpr) -> None:
        """
        assign C++ identifiers to all GExpr nodes
        collect the names of variables from GVarRef nodse
        """
        self.ident[ge] = 'f%d' % len(self.ident)

        if isinstance(ge, GVarRef):
            self.varids.add(ge.vname)

        if isinstance(ge, GFuncRef):
            self.gfs.add((ge.fname, len(ge.fargs)))

        if isinstance(ge, GInternal):
            for c in ge.children:
                self.walk_ast(c)

    def emit_sampler(self, mode: EmitMode) -> None:
        if mode == 'declaration':
            self.emit('\n\n// sampler declaration', trim=False)
            with self.indentation(f'''\
                class {self.sampler_name} : public {self.impl_name} {{
            ''', '};'):
                self.emit(f'''\
                    public:
                    using base_type = {self.impl_name};
                    using typename base_type::sample_type;
                    
                ''', trim=False)

                rules = ','.join(sorted(self.grammar.ruledefs.keys()))
                varids = ','.join(sorted(self.varids))

                self.emit(f'''\
                    enum class rule_t {{{rules}}};
                    enum class varid_t {{{varids}}};

                    // wrappers to convert type
                    sample_type get_var(varid_t varid) {{
                        return base_type::base_type::get_var(static_cast<int>(varid));
                    }}
                    void set_var(varid_t varid, const sample_type &value) {{
                        return base_type::base_type::set_var(static_cast<int>(varid), value);
                    }}
                ''')

                self.emit('\n/* === nodes === */', trim=False)
                for ruledef in self.grammar.ruledefs.values():
                    self.emit_method(ruledef.rhs, mode='declaration')

                self.emit('\n/* === ruledefs === */', trim=False)
                for ruledef in self.grammar.ruledefs.values():
                    self.emit(f'sample_type {ruledef.rname}();')

        elif mode == 'definition':
            self.emit('\n\n// sampler definition', trim=False)
            class_name = self.sampler_name

            for ruledef in self.grammar.ruledefs.values():
                self.emit_method(ruledef.rhs, mode='definition')

            for ruledef in self.grammar.ruledefs.values():
                with self.indentation(f'''\
                    // RuleDef {ruledef.locstr()}
                    inline {class_name}::sample_type {class_name}::{ruledef.rname}() {{
                ''', '}'):
                    self.emit(f'''
                        return {self.invoke(ruledef.rhs)};
                    ''')

    def emit_method(self, ge: GExpr, mode: EmitMode) -> None:
        # emit children first
        if isinstance(ge, GInternal):
            for c in ge.children:
                self.emit_method(c, mode)

        # skip types that are only ever directly invoked
        if isinstance(ge, (GTok, GCode, GDFuncRef, GFuncRef, GRuleRef, GVarRef)):
            return

        if mode == 'declaration':
            self.emit(f'sample_type {self.ident[ge]}();')
        elif mode == 'definition':
            handler_name = 'emit_method_' + ge.__class__.__name__
            m = cast(Optional[Callable[[GExpr], None]], getattr(self, handler_name, None))
            if m is None:
                msg = f'missing handler in {self.__class__.__name__}: {handler_name}'
                log.error(msg)
                raise CppEmitterError(msg)
            m(ge)

    def emit_method_GRep(self, ge: GRep) -> None:
        gid = self.ident[ge]
        with self.indentation(f'''\
            // GRep {ge.locstr()}
            inline {self.sampler_name}::sample_type {self.sampler_name}::{gid}() {{
        ''', '}'):
            self.emit(f'''\
                int lo={self.invoke_rep_bound(ge.lo, 0)};
                int hi={self.invoke_rep_bound(ge.hi, 2 ** 10)};

                int n={self.invoke_rep_dist(ge.dist)};

                sample_type s;
                while(n-->0) {{
                    s=cat(s,{self.invoke(ge.child)}); 
                }}
                return s;
            ''')

    def emit_method_GCat(self, ge: GCat) -> None:
        gid = self.ident[ge]
        with self.indentation(f'''\
            // GCat {ge.locstr()}
            inline {self.sampler_name}::sample_type {self.sampler_name}::{gid}() {{
        ''', '}'):
            self.emit(f'sample_type s={self.invoke(ge.children[0])};')
            for c in ge.children[1:]:
                self.emit(f's=cat(s,{self.invoke(c)});')
            self.emit(f'return s;')

    def emit_method_GAlt(self, ge: GAlt) -> None:
        gid = self.ident[ge]
        if ge.dynamic:
            dynamic_weights = ','.join(
                f'static_cast<double>({self.invoke(w)})' if isinstance(w, GCode) else str(w) for w in ge.weights
            )
            weights = '{' + dynamic_weights + '}'
        else:
            fixed_weights = ','.join(str(w) for w in cast(List[GTok], ge.weights))
            weights = '{' + fixed_weights + '}'

        with self.indentation(f'''\
            // GAlt {ge.locstr()}
            inline {self.sampler_name}::sample_type {self.sampler_name}::{gid}() {{
        ''', '}'):
            self.emit(f'''\
                int i = random.weighted_select({weights});
            ''')
            with self.indentation('switch(i) {', '}'):
                for i, c in enumerate(ge.children):
                    self.emit(f'''
                       case {i}:
                         return {self.invoke(c)};
                   ''')

            self.emit('return {}; // throw exception?')

    def emit_method_GTern(self, ge: GTern) -> None:
        gid = self.ident[ge]

        with self.indentation(f'''\
            // GTern {ge.locstr()}
            inline {self.sampler_name}::sample_type {self.sampler_name}::{gid}() {{
        ''', '}'):
            self.emit(f'''\
                return ({self.invoke(ge.code)})?({self.invoke(ge.children[0])}):({self.invoke(ge.children[1])});
            ''')

    def emit_method_GChooseIn(self, ge: GChooseIn) -> None:
        gid = self.ident[ge]

        with self.indentation(f'''\
            // GChooseIn {ge.locstr()}
            inline {self.sampler_name}::sample_type {self.sampler_name}::{gid}() {{
        ''', '}'):
            self.emit(f'''\
                push_vars();
            ''')
            for name, value_dist in zip(ge.vnames, ge.dists):
                self.emit(f'''\
                set_var(varid_t::{name}, {self.invoke(value_dist)});
                ''')
            self.emit(f'''\
                sample_type result = {self.invoke(ge.child)};
                pop_vars();
                return result;
            ''')

    def emit_method_GRange(self, ge: GRange) -> None:
        gid = self.ident[ge]

        # TODO we only handle multibyte mode.. "wide" chars are converted to multibyte with ENCODING
        encoded = [c.encode(ENCODING) for c in ge.chars]
        if any(len(c) > 1 for c in encoded):
            chars = ','.join(encode_as_cpp_str(c) for c in ge.chars)
            with self.indentation(f'''\
                // GRange {ge.locstr()}
                inline {self.sampler_name}::sample_type {self.sampler_name}::{gid}() {{
            ''', '}'):
                self.emit(f'''\
                    return random.choice({{{chars}}});
                ''')
        else:
            chars = ','.join(encode_as_cpp_char(c) for c in ge.chars)
            with self.indentation(f'''\
                // GRange {ge.locstr()}
                inline {self.sampler_name}::sample_type {self.sampler_name}::{gid}() {{
            ''', '}'):
                self.emit(f'''\
                    return {{1, random.choice({{{chars}}})}};
                ''')

    def emit_method_GDenoted(self, ge: GDenoted) -> None:
        gid = self.ident[ge]

        with self.indentation(f'''\
            // GDenoted {ge.locstr()}
            inline {self.sampler_name}::sample_type {self.sampler_name}::{gid}() {{
        ''', '}'):
            self.emit(f'''\
                return denote({self.invoke(ge.left)}, {self.invoke(ge.right)});
            ''')

    def invoke(self, ge: GExpr) -> str:
        """
        return a C++ expression that samples the given GExpr when executed.
        """
        if isinstance(ge, GRuleRef):
            return f'{ge.rname}()'
        elif isinstance(ge, GFuncRef):
            args = ','.join(self.as_gfunc_arg(c) for c in ge.fargs)
            return f'{ge.fname}({args})'
        elif isinstance(ge, GVarRef):
            return f'get_var(varid_t::{ge.vname})'
        elif isinstance(ge, GCode):
            return ge.expr
        elif isinstance(ge, GTok):
            if ge.type == 'string':
                return encode_as_cpp_str(ge.as_str())
            return str(ge)  # ints and floats
        else:
            return f'{self.ident[ge]}()'

    def as_gfunc_arg(self, ge: GExpr) -> str:
        """
        return a C++ callable to pass into a gfunc.

        Generated gfunc arguments are functions. When a gexpr isn't a function
        or method, wrap in a lambda.
        """
        if isinstance(ge, GTok):
            return f'[]() {{return {self.invoke(ge)};}}'
        elif isinstance(ge, GFuncRef):
            return f'[]() {{return {self.invoke(ge)};}}'
        elif isinstance(ge, GVarRef):
            return f'[]() {{return {self.invoke(ge)};}}'
        elif isinstance(ge, GCode):
            return f'[&]() {{return {self.invoke(ge)};}}'
        elif isinstance(ge, GRuleRef):
            # return f'std::bind(&{ge.rname}, this)'
            return f'[this]() -> sample_type {{return {ge.rname}();}}'
        else:
            # return f'(std::bind(&{self.ident[ge]}, this))'
            return f'[this]() -> sample_type {{return {self.ident[ge]}();}}'

    def invoke_rep_bound(self, x: Union[GTok, GCode, None], default: int) -> str:
        if x is None:
            return str(default)
        elif isinstance(x, GTok):
            return str(x)  # use value as written
        elif isinstance(x, GCode):
            return self.invoke(x)

    @staticmethod
    def invoke_rep_dist(dist: RepDist) -> str:
        """
        generate C++ source to sample the given RepDist.

        assume C++ variables lo, hi, and rand are available
        """
        fstr = 'std::min(std::max(lo, {0}), hi)'
        if dist.name.startswith('unif'):
            return 'random.uniform(lo,hi)'
        elif dist.name.startswith('geom'):
            # "a"{geom(n)} has an average of n copies of "a"
            parm = 1 / float(dist.args[0].as_num() + 1)
            return fstr.format(f'(random.geometric({parm})-1)')
        elif dist.name.startswith('norm'):
            args = ','.join(str(x) for x in dist.args)
            return fstr.format(f'static_cast<int>(random.normal({args})+.5)')
        elif dist.name.startswith('binom'):
            args = ','.join(str(x) for x in dist.args)
            return fstr.format(f'random.binomial({args})')
        elif dist.name == 'choose' or dist.name == 'choice':
            args = ','.join(str(x) for x in dist.args)
            return fstr.format(f'random.choice({{{args}}})')
        else:
            raise GrammaParseError('unknown repdist %s' % dist.name)

    def emit_implementation(self, mode: EmitMode) -> None:
        if mode == 'declaration':
            self.emit('\n\n// implementation declaration', trim=False)
            with self.indentation(f'''\
                using denotation_t = int;
                using sample_t = gramma::basic_sample<denotation_t, char>;

                class {self.impl_name}: public gramma::SamplerBase<{self.sampler_name}, sample_t> {{
            ''', '};'):
                self.emit(f'''\
                    public:
                    using base_type = gramma::SamplerBase<{self.sampler_name}, sample_t>;

                    // sampler API
                    sample_t cat(const sample_t &a, const sample_t &b) {{
                        return a+b;
                    }}
                    sample_t denote(const sample_t &a, const denotation_t &b) {{
                        return sample_t(a,b);
                    }}

                    // === state variables ===

                    // === gfuncs and dfuncs ===
                    sample_t show_den(sample_t a) {{
                        return sample_t(a + "<" + std::to_string(a.d) + ">", a.d);
                    }}

                    sample_t show_den_lazy(func_type m) {{
                        auto a=m();
                        return sample_t(a + "<" + std::to_string(a.d) + ">", a.d);
                    }}
                ''')

                # emit gfunc stubs
                skip = set(['show_den', 'show_den_lazy'])
                for fname, nargs in self.gfs:
                    if fname not in skip:
                        args = ','.join('func_type arg%d' % i for i in range(nargs))
                        self.emit(f'sample_type {fname}({args});')

        elif mode == 'definition':
            # emit gfunc stub definitions
            self.emit('\n\n// implementation definition', trim=False)
            class_name = self.impl_name
            skip = set(['show_den', 'show_den_lazy'])
            for fname, nargs in self.gfs:
                if fname not in skip:
                    args = ','.join('func_type arg%d' % i for i in range(nargs))
                    with self.indentation(f'{class_name}::sample_type {class_name}::{fname}({args}){{', '}'):
                        self.emit('return {};')

    def write_monolithic_main(self, count: Optional[int] = 1, seed: Optional[int] = 1,
                              out: Union[None, str, IO[str]] = None,
                              mode: str = 'w', close: bool = True) -> None:
        if out is not None:
            self.write_to(out, mode)

        self.emit(f'''\
            #include <string>
            #include <iostream>
            #include <utility>
            
            #include "gramma/gramma.hpp"
            #include "gramma/sample.hpp"
            
            class {self.sampler_name};
        ''', trim=False)

        self.emit_implementation('declaration')
        self.emit_sampler('declaration')
        self.emit_implementation('definition')
        self.emit_sampler('definition')
        self.emit_main(count, seed)
        if close:
            self.close()

    def emit_main(self, count: Optional[int], seed: Optional[int] = 1) -> None:
        self.emit('\n\n// entry point', trim=False)
        if seed is not None:
            seedstr = f'sampler.random.set_seed({seed});'
        else:
            seedstr = ''

        self.emit(f'''\
            int main() {{
                {self.sampler_name} sampler={self.sampler_name}();
                {seedstr}
                for({';;' if count is None else 'int n=0;n<' + str(count) + ';++n'}) 
                    std::cout << sampler.start();
                return 0;
            }}
        ''')


def main():
    import argparse
    from argparse import RawTextHelpFormatter

    dummy_arg_value = (None,)

    parser = argparse.ArgumentParser(description='generate a C++ sampler for a gramma GLF file',
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('glf', metavar='GLF_IN', type=argparse.FileType(),
                        help='input GLF file')
    parser.add_argument('-s', '--sampler-name', dest='sampler_name', metavar='SAMPLER_NAME', default=None, type=str,
                        help='Name to use for sampler class (default is based on GLF)')
    parser.add_argument('-i', '--impl-name', dest='impl_name', metavar='IMPL_NAME', default=None, type=str,
                        help='Name to use for implementation class (default is based on sampler-name)')

    parser.add_argument('-f', '--force', dest='force', action='store_true', default=False,
                        help='force writing of sampler if it already exists')

    parser.add_argument('-m', '--main', dest='main', action='store_true', default=False,
                        help='add main (and build an executable instead of a library')

    parser.add_argument('-b', '--bin', dest='bin_path', metavar='BIN', type=str,
                        nargs='?', const=dummy_arg_value, default=None,
                        help='executable (exe or library) to build after generating C++ (default name based on GLF)')

    args = parser.parse_args()

    logging.basicConfig(format="glf2cpp [%(levelname)s]: %(message)s", level=logging.INFO)

    glf_path = args.glf.name
    glf_path_less_extension = glf_path

    if glf_path_less_extension.lower().endswith('.glf'):
        glf_path_less_extension = glf_path_less_extension[:-4]

    # compute sampler_name
    if args.sampler_name is None:
        sampler_name = os.path.basename(glf_path_less_extension) + '_sampler'
        log.info(f'using --sampler-name {sampler_name}')
    else:
        sampler_name = args.sampler_name

    # compute impl_name
    if args.impl_name is None:
        impl_name = os.path.basename(glf_path_less_extension) + '_sampler'
        log.info(f'using --impl-name {impl_name}')
    else:
        impl_name = args.impl_name

    # compute grammar
    glf_text = args.glf.read()
    grammar = GrammaGrammar(glf_text)

    sampler_decl_path = sampler_name + '_decl.inc'
    sampler_def_path = sampler_name + '_def.inc'
    sampler_path = sampler_name + '.cpp'

    impl_path = sampler_name + '.cpp'

    emitter = CppEmitter(grammar, sampler_name)

    with emitter.write_to(sampler_decl_path):
        emitter.emit_sampler('declaration')

    with emitter.write_to(sampler_def_path):
        emitter.emit_sampler('definition')

    writing = True
    if os.path.exists(sampler_path):
        if args.force:
            log.info(f'forced overwrite of {sampler_path}')
        else:
            log.info(f'{sampler_path} exists, not overwriting')
            writing = False

    if writing:
        with emitter.write_to(sampler_path):
            emitter.emit(f'''\
                #include <string>
                #include <iostream>
                #include <utility>

                #include "gramma/gramma.hpp"
                #include "gramma/sample.hpp"

                class {sampler_name};
            ''')
            emitter.emit_implementation('declaration')
            emitter.emit(f'''\
                #include "{sampler_decl_path}"
            ''')
            emitter.emit_implementation('definition')
            emitter.emit(f'''\
                #include "{sampler_def_path}"
            ''')
            if args.main:
                emitter.emit_main(count=None, seed=None)

    build_bin = args.bin_path is not None

    if build_bin:
        import shutil
        import subprocess

        # compute bin_path
        if args.bin_path is dummy_arg_value:
            bin_path = sampler_name
            if sys.executable.lower().endswith('.exe'):
                if args.main:
                    bin_path += '.exe'
                else:
                    bin_path += '.dll'
            else:
                if not args.main:
                    bin_path += '.so'
            log.info(f'using --executable {bin_path}')
        else:
            bin_path = args.bin_path

        # get the compiler
        cxx = os.environ.get('CXX')
        if cxx is None:
            cxx = shutil.which('clang++') or shutil.which('g++')
        if cxx is None:
            log.error('no compiler found')
            sys.exit(2)

        call_args = [cxx, '-std=c++17', '-I', INCLUDE_DIR, '-o', bin_path, sampler_path]
        if not args.main:
            call_args.extend(['-c', '-shared'])

        result = subprocess.call(call_args)

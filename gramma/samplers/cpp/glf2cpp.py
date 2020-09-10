#!/usr/bin/env python3
r"""
C++ sampler support

TODO
- add rule wrapping for depth tracking
- support wchar_t
    - if we're in multibyte mode, use ENCODING, else, emit character to C++ file and use L'xxx' or L'\u....'
      with wchar_t
    - see emit_method_GRange

"""
from typing import Dict, cast, List, Union, Iterable, Optional, IO, Set, Callable, Tuple, Literal

import os
import sys
import re
import shutil

from gramma.util.emitters import Emitter
from ...parser import GExpr, GTok, GFuncRef, GInternal, GVarRef, GCode, GRuleRef, RepDist, \
    GrammaParseError, GAlt, GDenoted, GCat, GTern, GChooseIn, GRange, GRep, GrammaGrammar, GDFuncRef

import logging

log = logging.getLogger('gramma.samplers.cpp')

# TODO this is used to encode "wide" chars as UTF8 in multibyte mode
ENCODING = 'utf8'
DEFAULT_CHAR_TYPE: Literal['multibyte', 'wide'] = 'multibyte'

INCLUDE_DIR = os.path.join(os.path.dirname(__file__), 'include')

# generated code is only C++11
# for std::variant denotation_t, need C++17
# for auto parameter gdfuncs, need c++20
CXXFLAGS = ['-std=c++20']

EmitMode = Literal['definition', 'declaration']


def quote_shell_arg(s: str) -> str:
    if re.search(r'''[\s"']''', s) is not None:
        if '"' in s:
            s = s.replace('"', r'\"')
        return f'"{s}"'
    return s


def shell_join(call_args: List[str]) -> str:
    return ' '.join(quote_shell_arg(a) for a in call_args)


def get_compiler() -> Optional[str]:
    cxx = os.environ.get('CXX')
    if cxx is None:
        cxx = shutil.which('g++') or shutil.which('clang++')
    return cxx


def encode_as_cpp(s: Union[str, bytes, int], quote: str, char_type: Literal['multibyte', 'wide']) -> str:
    if char_type == 'wide':
        raise NotImplemented("wide string output hasn't been implemented yet")
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


def encode_as_cpp_str(s: Union[str, bytes], char_type: Literal['multibyte', 'wide']) -> str:
    return encode_as_cpp(s, quote='"', char_type=char_type)


def encode_as_cpp_char(c: str, char_type: Literal['multibyte', 'wide']) -> str:
    return encode_as_cpp(ord(c), quote="'", char_type=char_type)


class CppEmitterError(Exception):
    pass


# noinspection PyPep8Naming
class CppEmitter(Emitter):
    enforce_ltr: bool

    grammar: GrammaGrammar
    sampler_name: str
    impl_name: str

    nlocals: int
    ident: Dict[GExpr, str]
    varids: Set[str]  # (vname)
    gdfs: Set[Tuple[str, int]]  # (fname, nargs)
    gfs: Set[Tuple[str, int]]  # (fname, nargs)

    def __init__(self, grammar: GrammaGrammar, sampler_name: str,
                 enforce_ltr: bool = True,
                 out: Union[None, str, IO[str]] = None, echo: Optional[IO[str]] = None):
        Emitter.__init__(self, out, echo)

        self.enforce_ltr = enforce_ltr

        self.grammar = grammar
        self.sampler_name = sampler_name
        self.impl_name = sampler_name + '_impl'

        self.nlocals = 0

        # walk the AST to collect info
        self.ident = {}
        self.varids = set()
        self.gdfs = set()
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
        elif isinstance(ge, GDFuncRef):
            self.gdfs.add((ge.fname, len(ge.fargs)))
        elif isinstance(ge, GFuncRef):
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
                    parms = ','.join(f'sample_type {vname}' for vname in ruledef.params)
                    self.emit(f'sample_type {ruledef.rname}({parms});')

        elif mode == 'definition':
            class_name = self.sampler_name
            sample_type = f'{class_name}::sample_type'

            self.emit('\n\n// sampler definition', trim=False)
            for ruledef in self.grammar.ruledefs.values():
                self.emit_method(ruledef.rhs, mode='definition')

                parms = ','.join(f'{sample_type} {vname}' for vname in ruledef.params)
                with self.indentation(f'''\
                    // RuleDef {ruledef.locstr()}
                    inline {sample_type} {class_name}::{ruledef.rname}({parms}) {{
                ''', '}'):
                    if len(ruledef.params) > 0:
                        self.emit('''\
                            vars_guard_t _vars_guard=vars_guard();
                            ''')
                        for vname in ruledef.params:
                            self.emit(f'''\
                            set_var(varid_t::{vname}, {vname});
                            ''')
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
                int lo = {self.invoke_rep_bound(ge.lo, 0)};
                int hi = {self.invoke_rep_bound(ge.hi, 2 ** 10)};
                int n = lo==hi?lo:{self.invoke_rep_dist(ge)};

                sample_type s;
                while(n-->0) {{
                    icat(s,{self.invoke(ge.child)}); 
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
                self.emit(f'icat(s,{self.invoke(c)});')
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
                vars_guard_t _vars_guard=vars_guard();
            ''')
            for name, value_dist in zip(ge.vnames, ge.dists):
                self.emit(f'''\
                set_var(varid_t::{name}, {self.invoke(value_dist)});
                ''')
            self.emit(f'''\
                return {self.invoke(ge.child)};
            ''')

    def emit_method_GRange(self, ge: GRange) -> None:
        gid = self.ident[ge]

        # TODO handle multibyte mode by inserting L all over
        encoded = [c.encode(ENCODING) for c in ge.chars]
        if any(len(c) > 1 for c in encoded):
            chars = ','.join(encode_as_cpp_str(c, char_type=DEFAULT_CHAR_TYPE) for c in ge.chars)
            with self.indentation(f'''\
                // GRange {ge.locstr()}
                inline {self.sampler_name}::sample_type {self.sampler_name}::{gid}() {{
            ''', '}'):
                self.emit(f'''\
                    return random.choice({{{chars}}});
                ''')
        else:
            chars = ','.join(encode_as_cpp_char(c, char_type=DEFAULT_CHAR_TYPE) for c in ge.chars)
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
            if self.enforce_ltr:
                arg = self.next_local()
                self.emit(f'''\
                    auto &&{arg}={self.invoke(ge.left)};
                    return denote(std::move({arg}), {self.invoke(ge.right)});
                ''')
            else:
                self.emit(f'''\
                    return denote({self.invoke(ge.left)}, {self.invoke(ge.right)});
                ''')

    def next_local(self):
        a = f'a{self.nlocals}'
        self.nlocals += 1
        return a

    def invoke_function(self, name: str, gargs: List[GExpr]) -> str:
        if self.enforce_ltr and len(gargs) >= 2:
            argl = []
            for c in gargs[:-1]:
                arg = self.next_local()
                self.emit(f'auto &&{arg}={self.invoke(c)};')
                argl.append(f'std::move({arg})')
            argl.append(self.invoke(gargs[-1]))
        else:
            argl = [self.invoke(c) for c in gargs]
        return f'{name}({",".join(argl)})'

    def invoke(self, ge: GExpr) -> str:
        """
        return a C++ expression that samples the given GExpr when executed.
        """
        if isinstance(ge, GRuleRef):
            return self.invoke_function(ge.rname, ge.rargs)
        elif isinstance(ge, GDFuncRef):
            return self.invoke_function(ge.fname, ge.fargs)
        elif isinstance(ge, GFuncRef):
            args = ','.join(self.invoke_as_callable(c) for c in ge.fargs)
            return f'{ge.fname}({args})'
        elif isinstance(ge, GVarRef):
            return f'get_var(varid_t::{ge.vname})'
        elif isinstance(ge, GCode):
            return ge.expr
        elif isinstance(ge, GTok):
            if ge.type == 'string':
                return encode_as_cpp_str(ge.as_str(), char_type=DEFAULT_CHAR_TYPE)
            return str(ge)  # ints and floats
        else:
            return f'{self.ident[ge]}()'

    def invoke_as_callable(self, ge: GExpr) -> str:
        """
        return a C++ callable to pass into a gfunc.

        Generated gfunc arguments are functions. When a gexpr isn't a function
        or method, wrap in a lambda.
        """
        if isinstance(ge, GTok):
            return f'[]() {{return {self.invoke(ge)};}}'
        elif isinstance(ge, GFuncRef):
            return f'[this]() {{return {self.invoke(ge)};}}'
        elif isinstance(ge, GVarRef):
            return f'[this]() {{return {self.invoke(ge)};}}'
        elif isinstance(ge, GCode):
            return f'[&]() {{return {self.invoke(ge)};}}'
        elif isinstance(ge, GRuleRef):
            return f'[this]() {{return {self.invoke(ge)};}}'
        else:
            return f'[this]() -> sample_type {{return {self.ident[ge]}();}}'

    def invoke_rep_bound(self, x: Union[GTok, GCode, None], default: int) -> str:
        if x is None:
            return str(default)
        elif isinstance(x, GTok):
            return str(x)  # use value as written
        elif isinstance(x, GCode):
            return self.invoke(x)

    def invoke_rep_dist(self, ge: GRep) -> str:
        """
        generate C++ source to sample the given RepDist.

        assume lo,hi, and random are in scope
        """
        dist = ge.dist

        fstr = f'std::min(std::max(lo, {{0}}), hi)'
        if dist.name.startswith('unif'):
            return f'random.uniform(lo,hi)'
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
                using char_t = char;

                using string_t = std::basic_string<char_t>;
                // using char_t = wchar_t;

                struct denotation_t : public std::variant<int,double,string_t> {{
                    using base_type = std::variant<int,double,string_t>;
                    using base_type::variant;
                }};

                template<class T>
                string_t str(T && arg) {{
                    return std::to_string(std::forward<T>(arg));
                    //return std::to_wstring(std::forward<T>(arg));
                }};

                template<>
                string_t str<denotation_t &>(denotation_t &d) {{
                    switch(d.index()) {{
                        case 0: return str(std::get<int>(d));
                        case 1: return str(std::get<double>(d));
                        case 2: return std::get<string_t>(d);
                    }}
                    return {{}};
                }}

                using sample_t = gramma::basic_sample<denotation_t, char_t>;

                class {self.impl_name}: public gramma::SamplerBase<{self.sampler_name}, sample_t> {{
            ''', '};'):
                self.emit(f'''\
                    public:
                    using base_type = gramma::SamplerBase<{self.sampler_name}, sample_t>;
                    using trace_type = bool;

                    // sampler API
                    void icat(sample_t &a, const sample_t &b) {{
                        a+=b;
                    }}
                    sample_t denote(const sample_t &a, const denotation_t &b) {{
                        return sample_t(a,b);
                    }}

                    // gfuncs

                    // for testing
                    sample_t show_den_lazy(sample_factory_type m) {{
                        auto a=m();
                        return sample_t(a + "<" + str(a.d) + ">", a.d);
                    }}

                    // this non-lazy form of the previous gfunc relies on a
                    // callable-converting constructor of sample_t
                    sample_t show_den(sample_t a) {{
                        return sample_t(a + "<" + str(a.d) + ">", a.d);
                    }}

                    sample_t x2(sample_factory_type m) {{
                        auto a=m();
                        icat(a,a);
                        return a;
                    }}
                ''')

                # emit func stub declarations
                skip = {'show_den', 'show_den_lazy', 'x2'}
                for fname, nargs in self.gfs:
                    if fname not in skip:
                        args = ','.join('sample_factory_type arg%d' % i for i in range(nargs))
                        self.emit(f'sample_type {fname}({args});')
                self.emit('''
                    // gdfuncs
                    
                    // sample_t implicity passed via string_t variant constructor
                    denotation_t get_den(sample_t s)  {
                        return s.d;
                    }
                ''', trim=False)
                for fname, nargs in self.gdfs:
                    args = ','.join('denotation_type arg%d' % i for i in range(nargs))
                    self.emit(f'denotation_type {fname}({args});')

        elif mode == 'definition':
            # emit func stub definitions
            self.emit('\n\n// implementation definition', trim=False)
            class_name = self.impl_name
            skip = {'show_den', 'show_den_lazy', 'x2'}
            for fname, nargs in self.gfs:
                if fname not in skip:
                    args = ','.join('sample_factory_type arg%d' % i for i in range(nargs))
                    with self.indentation(f'{class_name}::sample_type {class_name}::{fname}({args}){{', '}'):
                        self.emit(f'return "({fname} stub)";')
            for fname, nargs in self.gdfs:
                args = ','.join('denotation_type arg%d' % i for i in range(nargs))
                with self.indentation(f'{class_name}::denotation_type {class_name}::{fname}({args}){{', '}'):
                    self.emit(f'return "({fname} stub)";')

    def emit_sampler_start(self) -> None:
        self.emit(f'''\
            #include <utility>
            #include <iostream>
            #include <string>
            #include <variant>

            #include "gramma/gramma.hpp"
            #include "gramma/sample.hpp"

            class {self.sampler_name};
        ''', trim=False)

    def write_monolithic_main(self, count: Optional[int] = 1, seed: Optional[int] = 1,
                              out: Union[None, str, IO[str]] = None,
                              mode: str = 'w', close: bool = True) -> None:
        if out is not None:
            self.write_to(out, mode)

        self.emit_sampler_start()
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


def check_force(path: str, force_level: int, force_limit: int) -> bool:
    if os.path.exists(path):
        if force_level >= force_limit:
            log.info(f'overwriting existing {path}')
            return True
        else:
            log.info(f'using existing {path}, overwrite with -{"f" * force_limit}')
            return False
    return True


def main(main_args: Optional[List[str]] = None) -> None:
    import argparse
    from argparse import RawTextHelpFormatter

    dummy_arg_value = (None,)

    parser = argparse.ArgumentParser(description='generate a C++ sampler for a gramma GLF file',
                                     formatter_class=RawTextHelpFormatter)

    parser.add_argument('glf', metavar='GLF_IN', type=argparse.FileType(),
                        help='input GLF file')

    parser.add_argument('--enforce-ltr', dest='enforce_ltr', action='store_true', default=False,
                        help='enforce left to right sampling of rule, denotation, and gdfunc arguments\n'
                             'note: to prescribe gfunc argument sample order, use sample_factory_type arguments')

    parser.add_argument('-s', '--sampler-name', dest='sampler_name', metavar='SAMPLER_NAME', default=None, type=str,
                        help='Name to use for sampler class (default is based on GLF)')

    parser.add_argument('-f', '--force', dest='force', action='count', default=0,
                        help='once to overwrite generated *.inc if it already exists\n'
                             'twice to overwrite generated files AND the implementation, *.cpp, where customizations '
                             'belong')

    parser.add_argument('-m', '--main', dest='main', action='store_true', default=False,
                        help='add main (and build an executable instead of a library')

    parser.add_argument('-b', '--bin', dest='bin_path', metavar='BIN', type=str,
                        nargs='?', const=dummy_arg_value, default=None,
                        help='executable (exe or library) to build after generating C++ (default name based on GLF)')

    args = parser.parse_args(args=main_args)

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

    # compute grammar
    glf_text = args.glf.read()
    grammar = GrammaGrammar(glf_text)
    args.glf.close()

    sampler_decl_path = sampler_name + '_decl.inc'
    sampler_def_path = sampler_name + '_def.inc'
    sampler_path = sampler_name + '.cpp'

    impl_path = sampler_name + '.cpp'

    emitter = CppEmitter(grammar, sampler_name, enforce_ltr=args.enforce_ltr)

    if check_force(sampler_decl_path, args.force, 1):
        with emitter.write_to(sampler_decl_path):
            emitter.emit_sampler('declaration')

    if check_force(sampler_def_path, args.force, 1):
        with emitter.write_to(sampler_def_path):
            emitter.emit_sampler('definition')

    if check_force(sampler_path, args.force, 5):
        with emitter.write_to(sampler_path):
            emitter.emit_sampler_start()
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
        import subprocess

        # compute bin_path
        if args.bin_path is dummy_arg_value:
            bin_path = sampler_name
            if sys.executable.lower().endswith('.exe'):  # pragma: no cover
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

        # get the compiler and build
        cxx = get_compiler()
        if cxx is None:
            log.error('no compiler found!')
        else:
            cmd_args = [cxx] + CXXFLAGS + ['-O3', '-I', INCLUDE_DIR, '-o', bin_path, sampler_path]
            if not args.main:
                cmd_args[1:1] = ['-fPIC', '-shared']

            log.info('  ' + shell_join(cmd_args))
            retcode = subprocess.call(cmd_args)

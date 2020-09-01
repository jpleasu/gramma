#!/usr/bin/env python3
"""
emit a C++ samplre class for GLF input

use crtp BaseXSampler with SampleClass and DenotationClass
parameters.

provide random api compatible with Python randomAPI


- for testing, we should be able to feed GLF, GExpr  in and get CPP out.
- gfunc stubs are generated seperately

- add gdfuncs and denotation parsing

- GVarRefs should be converted to an enum, which is used to perform lookups in vars : map<int, SampleT> instead of
    map<std::string, SampleT>

"""
import sys
import os
import re
from typing import Dict, cast, List, Union, Iterable, Optional, Generator, IO, Set, Callable

from ...parser import GExpr, GTok, GFuncRef, GInternal, GVarRef, GCode, GRuleRef, RepDist, \
    GrammaParseError, GAlt, RuleDef, GDenoted, GCat, GTern, GChooseIn, GRange, GRep, GrammaGrammar, log, GDFuncRef

from gramma.util.emitters import Emitter

# XXX more than this needs to change to change encoding..
ENCODING = 'utf8'


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
    ident: Dict[GExpr, str]
    varids: Set[str]

    grammar_name: str

    def __init__(self, out: IO[str], grammar: GrammaGrammar, grammar_name: str, echo: Optional[IO[str]] = None):
        Emitter.__init__(self, out, echo)

        self.grammar = grammar
        self.grammar_name = grammar_name

        # assign c++ identifiers to GExprs
        self.ident = {}
        self.varids = set()
        for ruledef in self.grammar.ruledefs.values():
            self.assign_ids(ruledef.rhs)

    def assign_ids(self, ge: GExpr) -> None:
        """
        assign C++ identifiers to GExprs
        """
        # self.ident[ge] = 'f%x' % id(ge)
        self.ident[ge] = 'f%d' % len(self.ident)

        if isinstance(ge, GVarRef):
            self.varids.add(ge.vname)

        if isinstance(ge, GInternal):
            for c in ge.children:
                self.assign_ids(c)

    def emit_base(self) -> None:
        sampler_base_name = self.grammar_name + '_base'
        include_path = os.path.join(sys.prefix, 'include')
        with self.indentation(f'''\
            /* build with
                -I {include_path} 
            */
            #include "gramma/gramma.hpp"

            template<class ImplT, class SampleT, class DenotationT>
            class {sampler_base_name} : public gramma::SamplerBase<ImplT, SampleT, DenotationT> {{
        ''', '};'):
            self.emit('''\
                public:
                using gramma::SamplerBase<ImplT,SampleT,DenotationT>::impl;
                
                using gramma::SamplerBase<ImplT,SampleT,DenotationT>::random;
                
                using gramma::SamplerBase<ImplT,SampleT,DenotationT>::set_var;
                using gramma::SamplerBase<ImplT,SampleT,DenotationT>::get_var;
                using gramma::SamplerBase<ImplT,SampleT,DenotationT>::push_vars;
                using gramma::SamplerBase<ImplT,SampleT,DenotationT>::pop_vars;
                
                using gramma::SamplerBase<ImplT,SampleT,DenotationT>::cat;
                using gramma::SamplerBase<ImplT,SampleT,DenotationT>::denote;
            ''')

            rules = ','.join(sorted(self.grammar.ruledefs.keys()))
            varids = ','.join(sorted(self.varids))

            self.emit(f'''\
                enum class rule_t {{{rules}}};
                enum class varid_t {{{varids}}};

                SampleT get_var(varid_t varid) {{
                    return get_var(static_cast<int>(varid));
                }}
                void set_var(varid_t varid, const SampleT &value) {{
                    return set_var(static_cast<int>(varid), value);
                }}
            ''')

            self.emit('/* === nodes === */')
            for ruledef in self.grammar.ruledefs.values():
                self.emit_method(ruledef.rhs)

            self.emit('/* === ruledefs === */')
            for ruledef in self.grammar.ruledefs.values():
                loc = ruledef.location
                locstr = '' if loc is None else f'line {loc.line}, column {loc.column}'
                with self.indentation(f'''\
                    // ruledef {locstr}
                    SampleT {ruledef.rname}() {{
                ''', '}'):
                    self.emit(f'''
                        return {self.invoke(ruledef.rhs)};
                    ''')

    def emit_method(self, ge: GExpr) -> None:
        """
        emit a c++ method that samples ge
        """

        # emit children first
        if isinstance(ge, GInternal):
            for c in ge.children:
                self.emit_method(c)

        # skip types that are only every directly invoked
        if isinstance(ge, (GTok, GCode, GDFuncRef, GFuncRef, GRuleRef, GVarRef)):
            return

        handler_name = 'emit_method_' + ge.__class__.__name__
        m = cast(Optional[Callable[[GExpr], None]], getattr(self, handler_name, None))
        if m is None:
            msg = f'missing handler in {self.__class__.__name__}: {handler_name}'
            log.error(msg)
            raise CppEmitterError(msg)
        return m(ge)

    def emit_method_GRep(self, ge: GRep) -> None:
        gid = self.ident[ge]
        with self.indentation(f'''\
            // GRep {ge.locstr()}
            SampleT {gid}() {{
        ''', '}'):
            self.emit(f'''\
                int lo={self.invoke_rep_bound(ge.lo, 0)};
                int hi={self.invoke_rep_bound(ge.hi, 2 ** 10)};

                int n={self.invoke_rep_dist(ge.dist)};

                SampleT s;
                while(n-->0) {{
                    s=cat(s,{self.invoke(ge.child)}); 
                }}
                return s;
            ''')

    def emit_method_GCat(self, ge: GCat) -> None:
        gid = self.ident[ge]
        with self.indentation(f'''\
            // GCat {ge.locstr()}
            SampleT {gid}() {{
        ''', '}'):
            self.emit(f'SampleT s={self.invoke(ge.children[0])};')
            for c in ge.children[1:]:
                self.emit(f's=cat(s,{self.invoke(c)});')
            self.emit(f'return s;')

    def emit_method_GAlt(self, ge: GAlt) -> None:
        gid = self.ident[ge]

        if ge.dynamic:
            weights = ','.join(
                f'static_cast<double>({self.invoke(w)})' if isinstance(w, GCode) else str(w) for w in ge.weights
            )
            weights = '{' + weights + '}'
        else:
            fixed_weights = ','.join(str(w) for w in cast(List[GTok], ge.weights))
            self.emit(f'''
                static constexpr double {gid}_weights[] {{{fixed_weights}}};
            ''')
            weights = f'{gid}_weights'

        with self.indentation(f'''\
            // GAlt {ge.locstr()}
            SampleT {gid}() {{
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
            SampleT {gid}() {{
        ''', '}'):
            self.emit(f'''\
                return ({self.invoke(ge.code)})?({self.invoke(ge.children[0])}):({self.invoke(ge.children[1])});
            ''')

    def emit_method_GChooseIn(self, ge: GChooseIn) -> None:
        gid = self.ident[ge]

        with self.indentation(f'''\
            // GChooseIn {ge.locstr()}
            SampleT {gid}() {{
        ''', '}'):
            self.emit(f'''\
                push_vars();
            ''')
            for name, value_dist in zip(ge.vnames, ge.dists):
                self.emit(f'''\
                    set_var(varid_t::{name}, {self.invoke(value_dist)});
                ''')
            self.emit(f'''\
                SampleT result = {self.invoke(ge.child)};
                pop_vars();
                return result;
            ''')

    def emit_method_GRange(self, ge: GRange) -> None:
        gid = self.ident[ge]

        encoded = [c.encode(ENCODING) for c in ge.chars]
        if any(len(c) > 1 for c in encoded):
            chars = ','.join(encode_as_cpp_str(c) for c in ge.chars)
            with self.indentation(f'''\
                // GRange {ge.locstr()}
                static constexpr char const *{gid}_chars[] {{{chars}}};
                SampleT {gid}() {{
            ''', '}'):
                self.emit(f'''\
                    return random.choice({gid}_chars);
                ''')
        else:
            chars = ','.join(encode_as_cpp_char(c) for c in ge.chars)
            with self.indentation(f'''\
                // GRange {ge.locstr()}
                static constexpr char {gid}_chars[] {{{chars}}};
                SampleT {gid}() {{
            ''', '}'):
                self.emit(f'''\
                    return {{1, random.choice({gid}_chars)}};
                ''')

    def emit_method_GDenoted(self, ge: GDenoted) -> None:
        gid = self.ident[ge]

        with self.indentation(f'''\
            // GDenoted {ge.locstr()}
            SampleT {gid}() {{
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
            return f'impl().{ge.fname}({args})'
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
            return f'std::bind(&ImplT::{ge.rname}, this)'
        else:
            return f'std::bind(&ImplT::{self.ident[ge]}, this)'

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

    def emit_simple_main(self, count: Optional[int] = 1, seed: int = 1) -> None:
        self.emit('''\
            #include <string>
            #include <iostream>
            #include <utility>
        ''')

        self.emit_base()

        self.emit(f'''\

            // customize sample and denotation types

            using denotation_t = int;

            struct sample_t:  public std::string {{
                denotation_t d;

                template<class T>
                sample_t(T s, denotation_t d=0) : std::string(std::forward<T>(s)), d(d) {{
                }}
                sample_t(int count, char c, denotation_t d=0) : std::string(count, c), d(d) {{
                }}

                sample_t() = default;
            }};


            // a sampler
            class {self.grammar_name}: public {self.grammar_name}_base<{self.grammar_name}, sample_t, denotation_t> {{
                public:
                sample_t cat(const sample_t &a, const sample_t &b) {{
                    return a+b;
                }}
                sample_t denote(const sample_t &a, const denotation_t &b) {{
                    return sample_t(a,b);
                }}

                sample_t show_den(method_t ag) {{
                    sample_t a = ag();
                    return sample_t(a + "<" + std::to_string(a.d) + ">", a.d);
                }}
            }};

            // and main
            int main() {{
                auto s={self.grammar_name}();
                s.random.set_seed({seed});
                for({';;' if count is None else 'int n=0;n<' + str(count) + ';++n'}) 
                    std::cout << s.start();
                return 0;
            }}
        ''')


# noinspection PyMethodMayBeStatic
class CppGen(Emitter):
    sampler_classname: str
    grammar: GrammaGrammar
    ident: Dict[GExpr, str]

    @property
    def ruledefs(self) -> Dict[str, RuleDef]:
        return self.grammar.ruledefs

    def __init__(self, glf_file, cpp_file, sampler_classname=None, impl_include_file=None):
        """
        create and emit a C++ implementation of a gramma sampler.
            glf_file                input GLF grammar file object
            cpp_file                output file object
            sampler_classname       C++ class name
            impl_include_file       C++ source to include in the sampler class that implements
                                    gfuncs, state variables, and the sideeffect API
        """
        Emitter.__init__(self, cpp_file)
        basename = os.path.basename(glf_file.name)
        if basename.endswith('.glf'):
            basename = basename[:-4]
        if sampler_classname is None:
            sampler_classname = basename.capitalize()
            if not re.search('sampler$', sampler_classname, re.IGNORECASE):
                sampler_classname += 'Sampler'
        self.sampler_classname = sampler_classname

        glf_source = glf_file.read()
        self.grammar = GrammaGrammar(glf_source)

        self.ident = {}
        for ruledef in self.ruledefs.values():
            self.assign_ids(ruledef.rhs)

        # extract gfunc call patterns? contexts

        if impl_include_file is None:
            impl_include_file = basename + '_sampler_impl.cpp'

        if not os.path.exists(impl_include_file):
            with open(impl_include_file, 'w') as out:
                impl = Emitter(out)
                impl.emit('''\
                    /* === declare state variables === */
                    
                    // for g4toglf generated grammars
                    int maxrep=3;
                    
                    // rule depth in trace tree
                    int rule_depth;

                    // set state prior to each sample
                    void reset_state() {
                        rule_depth=0;
                    }
                ''')
                with impl.indentation('#if defined(USE_SIDEEFFECT_API)', '#endif', flushleft=True, level=0):
                    impl.emit('''\
                        // XXX choose the type associated with each rule 
                        using sideeffect_t=bool;
                        
                        // XXX return the value associated with a rule just prior to its execution
                        sideeffect_t push(rule_t rule) {
                            ++rule_depth;
                            return true;
                        }
                        
                        // XXX handle the result immediately after rule execution
                        void pop(const sideeffect_t &assoc, const string_t &subsample) {
                            if(assoc) {
                                --rule_depth;
                            }
                        }
                        
                    ''')
                impl.emit('''\
                    // XXX define gfuncs
                    
                    // run method and return empty string.. mnemonic "execute"
                    template<typename T>
                    string_t e(T m) {
                        m();
                        return "";
                    }
                    
                ''')
                predefined_gfuncs = set('e'.split())
                gfs = set()  # pairs, (name, # args)
                for gf in self.gen_gfuncs():
                    gfs.add((gf.fname, len(gf.fargs)))
                for fname, nargs in sorted(gfs):
                    if fname in predefined_gfuncs:
                        continue
                    args = ','.join('method_t arg%d' % i for i in range(nargs))
                    with impl.indentation(f'string_t {fname}({args}){{', '}'):
                        impl.emit('return "?";')
        include_path = os.path.join(sys.prefix, 'include')
        with self.indentation(f'''\
            /* build with
                g++ -O3 -I {include_path} generated.cpp -o sampler
            */
            #include "gramma/gramma.hpp"

            // generated from {glf_file.name} 
            class {sampler_classname} : public gramma::SamplerBase {{
        ''', '};'):

            self.emit('public:')

            rules = ','.join(sorted(self.ruledefs.keys()))

            self.emit(f'''\
                    enum class rule_t {{{rules}}};

                   # include "{impl_include_file}"
               ''')

            self.emit('/* === nodes=== */')
            for ruledef in self.ruledefs.values():
                self.emit_method(ruledef.rhs)

            self.emit('/* === ruledefs=== */')
            for ruledef in self.ruledefs.values():
                with self.indentation(f'''\
                    // ruledef
                    string_t {ruledef.rname}() {{
                ''', '}'):
                    with self.indentation('#if defined(USE_SIDEEFFECT_API)', flushleft=True):
                        self.emit(f'''\
                            sideeffect_t assoc_value = push(rule_t::{ruledef.rname});
                            string_t subsample={self.invoke(ruledef.rhs)};
                            pop(assoc_value, subsample);
                            return subsample;
                        ''')
                    with self.indentation('#else', '#endif', flushleft=True):
                        self.emit(f'''
                            return {self.invoke(ruledef.rhs)};
                        ''')
        with self.indentation('#if defined(BUILDMAIN)', '#endif', flushleft=True):
            self.emit(f'''\
                int main() {{
                    {sampler_classname} sampler;

                    while(true) {{
                        sampler.reset_state();
                        std::cout << sampler.start();
                    }}

                    return 0;
                }}
            ''')

    def gen_gfuncs(self, ge: Optional[GExpr] = None) -> Generator[GFuncRef, None, None]:
        if ge is None:
            for ruledef in self.ruledefs.values():
                yield from self.gen_gfuncs(ruledef.rhs)
        if isinstance(ge, GFuncRef):
            yield ge
        if isinstance(ge, GInternal):
            for c in ge.children:
                yield from self.gen_gfuncs(c)

    def as_gfunc_arg(self, ge: GExpr) -> str:
        """
        Generate a C++ callable to pass into a gfunc.

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
            return f'std::bind(&{self.sampler_classname}::{ge.rname}, this)'
        else:
            return f'std::bind(&{self.sampler_classname}::{self.ident[ge]}, this)'

    def invoke(self, ge: GExpr) -> str:
        """
        genereate a C++ expression that samples the given GExpr when executed.
        """
        if isinstance(ge, GRuleRef):
            return f'{ge.rname}()'
        elif isinstance(ge, GFuncRef):
            args = ','.join(self.as_gfunc_arg(c) for c in ge.fargs)
            return f'{ge.fname}({args})'
        elif isinstance(ge, GVarRef):
            return f'get_var({encode_as_cpp_str(ge.vname)})'
        elif isinstance(ge, GCode):
            return ge.expr
        elif isinstance(ge, GTok):
            return encode_as_cpp_str(ge.as_str())
        else:
            return f'{self.ident[ge]}()'

    def invoke_rep_bound(self, x: Union[GTok, GCode, None], default: int) -> str:
        if x is None:
            return str(default)
        elif isinstance(x, GTok):
            return str(x)  # value as written
        elif isinstance(x, GCode):
            return self.invoke(x)

    def invoke_rep_dist(self, dist: RepDist) -> str:
        """
        generate C++ source to sample the given RepDist.

        assume C++ variables lo, hi, and rand are available
        """
        fstr = 'std::min(std::max(lo, {0}), hi)'
        if dist.name.startswith('unif'):
            return 'std::uniform_int_distribution<int>(lo,hi)(rand)'
        elif dist.name.startswith('geom'):
            # "a"{geom(n)} has an average of n copies of "a"
            parm = 1 / float(dist.args[0].as_num() + 1)
            return fstr.format(f'(std::geometric_distribution<int>({parm})(rand)-1)')
        elif dist.name.startswith('norm'):
            args = ','.join(str(x) for x in dist.args)
            return fstr.format(f'static_cast<int>(std::normal_distribution<double>({args})(rand)+.5)')
        elif dist.name.startswith('binom'):
            args = ','.join(str(x) for x in dist.args)
            return fstr.format(f'(std::binomial_distribution<int>({args})(rand))')
        elif dist.name.startswith('choose'):
            args = ','.join(str(x) for x in dist.args)
            return fstr.format(f'([](){{int arr[]={args};return uniform_selection(arr);}}())')
        else:
            raise GrammaParseError('unknown repdist %s' % dist.name)

    def emit_method(self, ge: GExpr) -> None:
        """
        emit a C++ method that samples the given GExpr when called.
        """
        # emit children first
        if isinstance(ge, GInternal):
            for c in ge.children:
                self.emit_method(c)

        gid = self.ident[ge]
        if isinstance(ge, GTok):
            pass  # invoked directly
        elif isinstance(ge, GAlt):
            def emit_cases(galt: GAlt = cast(GAlt, ge)) -> None:
                for i, c in enumerate(galt.children):
                    self.emit(f'''
                        case {i}:
                          return {self.invoke(c)};
                    ''')

            if ge.dynamic:
                weights = ','.join(
                    f'static_cast<double>({self.invoke(w)})' if isinstance(w, GCode) else str(w) for w in ge.weights
                )
                with self.indentation(f'''\
                    // GAlt dynamic {ge.locstr()}
                    string_t {gid}() {{
                ''', '}'):
                    self.emit(f'''\
                        int i = weighted_select({{{weights}}});
                    ''')
                    with self.indentation('switch(i) {', '}'):
                        emit_cases()
                    self.emit('return {}; // throw exception?')
            else:
                weights = ','.join(str(w) for w in cast(List[GTok], ge.weights))
                with self.indentation(f'''\
                    // GAlt static {ge.locstr()}
                    static constexpr double {gid}_weights[] {{{weights}}};
                    string_t {gid}() {{
                ''', '}'):
                    self.emit(f'''\
                        int i = weighted_select({gid}_weights);
                    ''')
                    with self.indentation('switch(i) {', '}'):
                        emit_cases()
                    self.emit('return {}; // throw exception?')
        elif isinstance(ge, GDenoted):
            with self.indentation(f'''\
                // gdenoted
                string_t {gid}() {{
            ''', '}'):
                self.emit(f'''\
                    string_t s = {self.invoke(ge.left)};
                    {self.invoke(ge.right)};
                    return s;
                ''')
        elif isinstance(ge, GCat):
            with self.indentation(f'''\
                // gcat
                string_t {gid}() {{
            ''', '}'):
                self.emit(f'string_t s={self.invoke(ge.children[0])};')
                for c in ge.children[1:]:
                    self.emit(f's+={self.invoke(c)};')
                self.emit(f'return s;')
        elif isinstance(ge, GRuleRef):
            pass  # invoked directly
        elif isinstance(ge, GFuncRef):
            pass  # invoked directly
        elif isinstance(ge, GTern):
            with self.indentation(f'''\
                // gtern
                string_t {gid}() {{
            ''', '}'):
                self.emit(f'''\
                    if({self.invoke(ge.code)}) {{
                        return {self.invoke(ge.children[0])};
                    }} else {{
                        return {self.invoke(ge.children[1])};
                    }}
                ''')
        elif isinstance(ge, GChooseIn):
            with self.indentation(f'''\
                // gchoosein
                string_t {gid}() {{
            ''', '}'):
                self.emit(f'''\
                    push_vars();
                ''')
                for name, value_dist in zip(ge.vnames, ge.dists):
                    self.emit(f'''\
                        set_var({encode_as_cpp_str(name)}, {self.invoke(value_dist)});
                    ''')
                self.emit(f'''\
                    string_t result = {self.invoke(ge.child)};
                    pop_vars();
                    return result;
                ''')
        elif isinstance(ge, GVarRef):
            pass  # invoked directly
        elif isinstance(ge, GCode):
            pass  # invoked directly
        elif isinstance(ge, GRange):
            encoded = [c.encode(ENCODING) for c in ge.chars]
            if any(len(c) > 1 for c in encoded):
                chars = ','.join(encode_as_cpp_str(c) for c in ge.chars)
                with self.indentation(f'''\
                    // grange (unicode)
                    static constexpr char const *{gid}_chars[] {{{chars}}};
                    string_t {gid}() {{
                ''', '}'):
                    self.emit(f'''\
                        return uniform_selection({gid}_chars);
                    ''')
            else:
                chars = ','.join(encode_as_cpp_char(c) for c in ge.chars)
                with self.indentation(f'''\
                    // grange
                    static constexpr char {gid}_chars[] {{{chars}}};
                    string_t {gid}() {{
                ''', '}'):
                    self.emit(f'''\
                        return string_t(1,uniform_selection({gid}_chars));
                    ''')
        elif isinstance(ge, GRep):
            with self.indentation(f'''\
                // grep
                string_t {gid}() {{
            ''', '}'):
                self.emit(f'''\
                    int lo={self.invoke_rep_bound(ge.lo, 0)};
                    int hi={self.invoke_rep_bound(ge.hi, 2 ** 10)};

                    int n={self.invoke_rep_dist(ge.dist)};

                    string_t s;
                    while(n-->0) {{
                        s+={self.invoke(ge.child)};
                    }}
                    return s;
                ''')
        else:
            with self.indentation(f'''\
                // placeholder[{ge.__class__.__name__}]
                string_t {gid}() {{
            ''', '}'):
                self.emit('''\
                    return "?";
                ''')

    def assign_ids(self, ge: GExpr) -> None:
        """
        assign C++ identifiers to GExprs
        """
        # self.ident[ge] = 'f%x' % id(ge)
        self.ident[ge] = 'f%d' % len(self.ident)

        if isinstance(ge, GInternal):
            for c in ge.children:
                self.assign_ids(c)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='generate C++ source for a generator from GLF')
    parser.add_argument('glf', metavar='GLF_IN',
                        type=argparse.FileType(), help='input GLF file')
    parser.add_argument('-o', '--out', dest='cpp', metavar='CPP_OUT',
                        type=argparse.FileType('w'), help='output C++ file', default='-')
    commandline_args = parser.parse_args()

    CppGen(commandline_args.glf, commandline_args.cpp)

# vim: ts=4 sw=4

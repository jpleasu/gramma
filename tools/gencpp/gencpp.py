#!/usr/bin/env python3
"""
emit C++ source for a function that generates samples from the GLF input file
"""
import os
import re
import textwrap

from gramma import *


def encode_as_cpp_str(s):
    r = '"'
    b = s.encode('utf8')
    for c in b:
        if c == 0:
            r += r'\0'
        elif c == 10:
            r += r'\n'
        elif c == 13:
            r += r'\r'
        elif c == 9:
            r += r'\t'
        elif c == 34:
            r += r'\"'
        elif c == 92:
            r += r'\\'  # two backslashes
        elif 0x20 <= c <= 0x7e:
            r += chr(c)
        else:
            r += f'\\x{c:02x}'
    return r + '"'


class IndentationContext(object):
    def __init__(self, emitter, pre, post, flushleft, tag, level):
        self.emitter = emitter
        self.pre = pre
        self.post = post
        self.flushleft = flushleft
        self.tag = tag
        self.level = level

    def __enter__(self):
        if self.pre is not None:
            self.emitter.emit(self.pre, flushleft=self.flushleft)
        self.emitter.indent += self.level

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.emitter.indent -= self.level
        if self.post is not None:
            self.emitter.emit(self.post, flushleft=self.flushleft, tag=self.tag)


class Emitter(object):
    """
    Utility for emitting formatted code.
    """

    def __init__(self, out):
        self.indent = 0
        self.out = out
        self.afters = {}

    def indentation(self, pre=None, post=None, flushleft=False, tag=None, level=1):
        return IndentationContext(self, pre, post, flushleft, tag, level)

    def emit(self, s, pre='', post='\n', trim=True, flushleft=False, after=None, tag=None):
        if after is not None:
            def closure():
                self.emit(s, pre=pre, post=post, trim=trim, flushleft=flushleft, tag=tag)

            self.afters.setdefault(after, []).append(closure)
            return

        s = textwrap.dedent(s)
        if trim:
            s = s.strip('\n')
        if not flushleft:
            s = textwrap.indent(s, '    ' * self.indent)
        self.out.write(pre + s + post)
        self.out.flush()
        if tag is not None:
            for closure in self.afters.pop(tag, []):
                closure()


class CppGen(Emitter):
    def __init__(self, glf_file, cpp_file, sampler_classname=None, fixed_methods_include=None):
        Emitter.__init__(self, cpp_file)
        basename = os.path.basename(glf_file.name)
        if basename.endswith('.glf'):
            basename = basename[:-4]
        if sampler_classname is None:
            sampler_classname = basename.capitalize()
            if not re.search('sampler$', sampler_classname, re.IGNORECASE):
                sampler_classname += 'Sampler'
        self.sampler_classname = sampler_classname
        if fixed_methods_include is None:
            fixed_methods_include = basename + '_sampler_impl.cpp'

        glf_source = glf_file.read()

        lt = GrammaGrammar.ruledef_parser.parse(glf_source)

        self.ruledefs = {}
        for ruledef in lt.children:
            lp = LarkTransformer()
            self.ruledefs[ruledef.children[0].value] = lp.visit(ruledef.children[1])

        self.ident = {}
        for ge in self.ruledefs.values():
            self.assign_ids(ge)

        # extract gfunc call patterns? contexts

        if not os.path.exists(fixed_methods_include):
            with open(fixed_methods_include, 'w') as out:
                impl=Emitter(out)
                impl.emit('''\
                        // XXX declare state variables

                        // XXX set state variables
                        void reset_state() {
                        }

                        // XXX define gfuncs
                    ''')
                gfs = set()  # pairs, (name, # arsg)
                for gf in self.gen_gfuncs():
                    gfs.add((gf.fname, len(gf.fargs)))
                for fname, nargs in sorted(gfs):
                    args = ','.join('method_type arg%d' % i for i in range(nargs))
                    with impl.indentation(f'string {fname}({args}){{', '}'):
                        impl.emit('return "?";')
        with self.indentation(f'''\
            #include "gramma.hpp"

            class {sampler_classname} : public gramma::SamplerBase {{
        ''', '};'):

            self.emit('public:')
            self.emit(f'''\
                   # include "{fixed_methods_include}"
               ''')

            self.emit('// nodes')
            for ge in self.ruledefs.values():
                self.dump(ge)

            self.emit('// ruledefs')
            for rname, ge in self.ruledefs.items():
                with self.indentation(f'''\
                    // ruledef
                    string {rname}() {{
                ''', '}'):
                    self.emit(f'''\
                        return {self.invoke(ge)};
                    ''')
        with self.indentation('#if defined(BUILDMAIN)', '#endif', flushleft=True):
            self.emit(f'''\
                int main() {{
                    {sampler_classname} sampler;
                    while(true) {{
                        std::cout << sampler.start();
                    }}
                    return 0;
                }}
            ''')

    def gen_gfuncs(self, ge=None):
        if ge is None:
            for ge in self.ruledefs.values():
                for gf in self.gen_gfuncs(ge):
                    yield gf
        if isinstance(ge, GFunc):
            yield ge
        if isinstance(ge, GInternal):
            for c in ge.children:
                for gf in self.gen_gfuncs(c):
                    yield gf

    def as_gfunc_arg(self, ge):
        """
        Generate a C++ callable to pass into a gfunc.

        Generated gfunc arguments are functions. When a gexpr isn't a function
        or method, wrap in a lambda.
        """
        if isinstance(ge, GTok):
            return f'[]() {{return {self.invoke(ge)};}}'
        elif isinstance(ge, GFunc):
            return f'[]() {{return {self.invoke(ge)};}}'
        elif isinstance(ge, GVar):
            return f'[]() {{return {self.invoke(ge)};}}'
        elif isinstance(ge, GRule):
            return f'std::bind(&{self.sampler_classname}::{ge.rname}, this)'
        else:
            return f'std::bind(&{self.sampler_classname}::{self.ident[ge]}, this)'

    def invoke(self, ge):
        if isinstance(ge, GRule):
            return f'{ge.rname}()'
        elif isinstance(ge, GFunc):
            args = ','.join(self.as_gfunc_arg(c) for c in ge.fargs)
            return f'{ge.fname}({args})'
        elif isinstance(ge, GVar):
            return f'get_var({encode_as_cpp_str(ge.vname)})'
        elif isinstance(ge, GCode):
            return ge.expr
        elif isinstance(ge, GTok):
            return encode_as_cpp_str(ge.as_str())
        else:
            return f'{self.ident[ge]}()'

    def invoke_rep_bound(self, x, default):
        if isinstance(x, int):
            return str(x)
        elif isinstance(x, GCode):
            return self.invoke(x)
        else:
            return str(default)

    def invoke_rep_dist(self, dist):
        """
        generate C++ source to randomly sample the given RepDist.

        assume C++ variables lo, hi, and rand are available
        """
        fstr = 'std::min(std::max(lo, {0}), hi)'
        if dist.name == 'unif':
            return 'std::uniform_int_distribution<int>(lo,hi)(rand)'
        elif dist.name == 'geom':
            # "a"{geom(n)} has an average of n copies of "a"
            parm = 1 / float(dist.args[0] + 1)
            return fstr.format(f'(std::geometric_distribution<int>({parm})(rand)-1)')
        elif dist.name == 'norm':
            args = ','.join(str(x) for x in dist.args)
            return fstr.format(f'static_cast<int>(std::normal_distribution<double>({args})(rand)+.5)')
        elif dist.name == 'binom':
            args = ','.join(str(x) for x in dist.args)
            return fstr.format(f'(std::binomial_distribution<int>({args})(rand))')
        elif dist.name == 'choose':
            args = ','.join(str(x) for x in dist.args)
            return fstr.format(f'([](){{int arr[]={args};return uniform_selection(arr);}}())')
        else:
            raise GrammaParseError('no dist %s' % (dist.name))

    def dump(self, ge):
        # emit children first
        if isinstance(ge, GInternal):
            for c in ge.children:
                self.dump(c)

        gid = self.ident[ge]
        if isinstance(ge, GTok):
            pass  # invoked directly
        elif isinstance(ge, GAlt):
            def emit_cases():
                for i, c in enumerate(ge.children):
                    self.emit(f'''
                        case {i}:
                          return {self.invoke(c)};
                    ''')

            if ge.dynamic:
                # compute weight expressions and normalize

                weights = ','.join(
                    f'static_cast<double>({self.invoke(w)})' if isinstance(w, GCode) else str(w) for w in ge.weights
                )
                with self.indentation(f'''\
                    // galt - dynamic
                    string {gid}() {{
                ''', '}'):
                    self.emit(f'''\
                        double weights[] {{{weights}}};
                        // normalize(weights);
                        int i = weighted_select(weights);
                    ''')
                    with self.indentation('switch(i) {', '}'):
                        emit_cases()
                    self.emit('return ""; // throw exception?')
            else:
                weights = ','.join(str(d) for d in ge.nweights)
                with self.indentation(f'''\
                    // galt
                    static constexpr double {gid}_weights[] {{{weights}}};
                    string {gid}() {{
                ''', '}'):
                    self.emit(f'''\
                        int i = weighted_select({gid}_weights);
                    ''')
                    with self.indentation('switch(i) {', '}'):
                        emit_cases()
                    self.emit('return ""; // throw exception?')
        elif isinstance(ge, GCat):
            with self.indentation(f'''\
                // gcat
                string {gid}() {{
            ''', '}'):
                # "" + "" + ...  -> string {""} + "" + ...
                if len(ge.children) >= 2 and isinstance(ge.children[0], GTok) and isinstance(ge.children[1], GTok):
                    argsum = 'string {' + self.invoke(ge.children[0]) + '} +' + '+'.join(
                        f'{self.invoke(c)}' for c in ge.children[1:])
                else:
                    argsum = '+'.join(f'{self.invoke(c)}' for c in ge.children)
                self.emit(f'return {argsum};')
        elif isinstance(ge, GRule):
            pass  # invoked directly
        elif isinstance(ge, GFunc):
            pass  # invoked directly
        elif isinstance(ge, GTern):
            with self.indentation(f'''\
                // gtern
                string {gid}() {{
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
                string {gid}() {{
            ''', '}'):
                self.emit(f'''\
                    push_vars();
                ''')
                for name, value_dist in zip(ge.vnames, ge.dists):
                    self.emit(f'''\
                        set_var({encode_as_cpp_str(name)}, {self.invoke(value_dist)});
                    ''')
                self.emit(f'''\
                    string result = {self.invoke(ge.child)};
                    pop_vars();
                    return result;
                ''')
        elif isinstance(ge, GVar):
            pass  # invoked directly
        elif isinstance(ge, GRange):
            chars = ','.join(f"'{c}'" for c in ge.chars)
            with self.indentation(f'''\
                // grange
                static constexpr char {gid}_chars[] {{{chars}}};
                string {gid}() {{
            ''', '}'):
                self.emit(f'''\
                    return string(1,uniform_selection({gid}_chars));
                ''')
        elif isinstance(ge, GRep):
            with self.indentation(f'''\
                // grep
                string {gid}() {{
            ''', '}'):
                self.emit(f'''\
                    int lo={self.invoke_rep_bound(ge.lo, 0)};
                    int hi={self.invoke_rep_bound(ge.hi, 2 ** 10)};

                    int n={self.invoke_rep_dist(ge.dist)};

                    string s;
                    while(n-->0) {{
                        s+={self.invoke(ge.child)};
                    }}
                    return s;
                ''')
        else:
            with self.indentation(f'''\
                // placeholder[{ge.__class__.__name__}]
                string {gid}() {{
            ''', '}'):
                self.emit('''\
                    return "?";
                ''')

    def assign_ids(self, ge):
        # self.ident[ge] = 'f%x' % id(ge)
        self.ident[ge] = 'f%d' % len(self.ident)

        if isinstance(ge, GInternal):
            for c in ge.children:
                self.assign_ids(c)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='generate a C++ gramma from GLF')
    parser.add_argument('glf', metavar='GLF_IN',
                        type=argparse.FileType('r'), help='input GLF file')
    parser.add_argument('-o', '--out', dest='cpp', metavar='CPP_OUT',
                        type=argparse.FileType('w'), help='output C++ file', default='-')
    args = parser.parse_args()

    CppGen(args.glf, args.cpp)

# vim: ts=4 sw=4
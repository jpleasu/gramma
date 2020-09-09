#!/usr/bin/env python3

import unittest
from io import StringIO
from typing import cast

from gramma.parser import GrammaGrammar, GFuncRef, GRuleRef, GChooseIn, GVarRef, GAlt, GRep, RepDist, GRange, GDenoted, \
    GTok, GCode, GDFuncRef, GrammaParseError, GCat, GTern, GExpr


# from IPython import embed;embed()


class TestExceptions(unittest.TestCase):
    def test_missing_variable(self):
        with self.assertRaises(GrammaParseError):
            GrammaGrammar('''
                start := x;
            ''')

    def test_grammar_of(self):
        with self.assertRaises(TypeError):
            GrammaGrammar.of(17)


class TestTokens(unittest.TestCase):

    def test_not_a_num(self):
        with self.assertRaises(GrammaParseError):
            GTok.from_str('astring').as_num()

    def test_no_string(self):
        self.assertEqual(GTok.from_int(17).as_str(), '(None)')

    def test_empty(self):
        e = GTok.new_empty()
        self.assertEqual(e.s, '')

    def test_join(self):
        self.assertEqual(GTok.join([GTok.from_str('a'), GTok.from_str('b')]).s, 'ab')

    def test_as_native(self):
        self.assertEqual(GTok.from_str('a').as_native(), "a")
        self.assertEqual(GTok.from_int(17).as_native(), 17)
        self.assertEqual(GTok.from_float(14.2).as_native(), 14.2)


class TestLarkParser(unittest.TestCase):
    """
    details of the Lark parsetree straight from the GLF grammar
    """

    def test_lark_choosein(self):
        lt = GrammaGrammar.GEXPR_PARSER.parse('''
            choose x~'a', y~'b', z~'c' in x.y.(choose x~'d' in x|z)
        ''')
        varnames = [c.children[0].value for c in lt.children[:-1:2]]
        self.assertEqual(len(varnames), 3)
        self.assertEqual(varnames, ['x', 'y', 'z'])
        bindings = [str(c.children[0].value) for c in lt.children[1:-1:2]]
        self.assertEqual(bindings, ["'a'", "'b'", "'c'"])
        expr = lt.children[-1]
        self.assertEqual(expr.data, 'cat')

        nested_choosein = expr.children[-1]
        self.assertEqual(nested_choosein.data, 'choosein')
        # nested variable names
        self.assertEqual([c.children[0].value for c in nested_choosein.children[:-1:2]], ['x'])
        # nested variable bindings
        self.assertEqual([c.children[0].value for c in nested_choosein.children[1:-1:2]], ["'d'"])
        # nested expression
        self.assertEqual(nested_choosein.children[-1].data, 'alt')

    def test_lark_func(self):
        lt = GrammaGrammar.GEXPR_PARSER.parse('''
            f('a','b',7,1.2,`expr`)
        ''')
        self.assertEqual(lt.data, 'func')
        self.assertEqual(lt.children[0].children[0].value, 'f')
        args = lt.children[1]
        self.assertEqual(args.data, 'func_args')
        self.assertEqual(args.children[0].data, 'string')
        self.assertEqual(args.children[1].data, 'string')
        self.assertEqual(args.children[2].data, 'number')
        self.assertEqual(args.children[2].children[0].type, 'INT')
        a = args.children[3]
        self.assertEqual(a.data, 'number')
        self.assertEqual(a.children[0].type, 'FLOAT')
        self.assertEqual(args.children[4].data, 'code')

    def test_lark_sampler(self):
        lt = GrammaGrammar.GLF_PARSER.parse('''
            r0:='a';
            r1:=choose x~'a' in 'b';
            r2:='a'|'b';
            r3:=`True`?'a':'b';
            r4:='a'/'b';
            r5:='a'.'b';
            r6:='a'{1,3};
            r7:=['a'..'b'];
            r8:=f('a',`x`,10,1.2);

            r9(a):= a/'b';
            r10(a):= a/f('b','c');
        ''')
        self.assertTrue(all(c.data == 'ruledef' for c in lt.children), msg='all children should be ruledefs')
        self.assertEqual(len(lt.children), 11, msg='wrong number of ruledefs')
        self.assertEqual(lt.children[0].children[1].data, 'string')
        self.assertEqual(lt.children[1].children[1].data, 'choosein')
        self.assertEqual(lt.children[2].children[1].data, 'alt')
        self.assertEqual(lt.children[3].children[1].data, 'tern')
        self.assertEqual(lt.children[4].children[1].data, 'denoted')
        self.assertEqual(lt.children[5].children[1].data, 'cat')
        self.assertEqual(lt.children[6].children[1].data, 'rep')
        self.assertEqual(lt.children[7].children[1].data, 'range')
        self.assertEqual(lt.children[8].children[1].data, 'func')

        self.assertTrue(all(len(c.children) == 2 for c in lt.children[:9]),
                        msg='non parameterized ruledefs should have 2 children')
        self.assertTrue(all(len(c.children) == 3 for c in lt.children[9:]),
                        msg='non parameterized ruledefs should have 2 children')

        r = lt.children[9]
        self.assertEqual(r.children[1].data, 'rule_parms')
        self.assertEqual(r.children[2].data, 'denoted')

        r = lt.children[10]
        self.assertEqual(r.children[1].data, 'rule_parms')
        d = r.children[2]
        self.assertEqual(d.data, 'denoted')
        self.assertEqual(d.children[1].data, 'dfunc')


class TestGrammaGrammar(unittest.TestCase):
    def test_factory(self):
        g = GrammaGrammar.of("start:='a';")
        self.assertEqual(str(g.ruledefs['start'].rhs), "'a'")

        fileobj = StringIO("start:='a';")
        g = GrammaGrammar.of(fileobj)
        self.assertEqual(str(g.ruledefs['start'].rhs), "'a'")

        g = GrammaGrammar.of(g)
        self.assertEqual(str(g.ruledefs['start'].rhs), "'a'")

    def test_ruledefs(self):
        g = GrammaGrammar('''
            r0:='a';
            r1:=choose x~'a' in 'b';
            r2:='a'|'b';
            r3:=`True`?'a':'b';
            r4:='a'/'b';
            r5:='a'.'b';
            r6:='a'{1,3};
            r7:=['a'..'b'];
            r8:=f('a',`x`,10,1.2);

            r9(a):= a/'b';
        ''')
        self.assertEqual(len(g.ruledefs), 10)
        self.assertEqual(str(g.ruledefs['r0'].rhs), """'a'""")
        self.assertEqual(str(g.ruledefs['r1'].rhs), """choose x~'a' in 'b'""")
        self.assertEqual(str(g.ruledefs['r2'].rhs), """'a'|'b'""")
        self.assertEqual(str(g.ruledefs['r3'].rhs), """`True`?'a':'b'""")
        self.assertEqual(str(g.ruledefs['r4'].rhs), """'a'/'b'""")
        self.assertEqual(str(g.ruledefs['r5'].rhs), """'a'.'b'""")
        self.assertEqual(str(g.ruledefs['r6'].rhs), """'a'{1,3}""")
        self.assertEqual(str(g.ruledefs['r7'].rhs), """['a'..'b']""")
        self.assertEqual(str(g.ruledefs['r8'].rhs), """f('a',`x`,10,1.2)""")
        self.assertEqual(str(g.ruledefs['r9'].rhs), """a/'b'""")

    def test_alt(self):
        g = GrammaGrammar('''
            r0 := 'a' | 2 'b' | 3.0 'c' | `4` 'd';
        ''')
        r0: GAlt = g.ruledefs['r0'].rhs
        self.assertIsInstance(r0, GAlt)
        self.assertEqual(r0.weights[0].as_int(), 1)
        self.assertEqual(r0.weights[1].as_int(), 2)
        self.assertEqual(r0.weights[2].as_float(), 3.0)
        self.assertEqual(r0.weights[3].expr, '4')

    def test_rep1(self):
        g = GrammaGrammar('''
        r0 := 'a'{1};
        r1 := 'a'{`x`};
        r2 := 'a'{,};
        r3 := 'a'{dist(1)};
    ''')
        for rulename, ruledef in g.ruledefs.items():
            self.assertIsInstance(ruledef.rhs, GRep)
            self.assertEqual(str(ruledef.rhs.child), "'a'")
        r0: GRep = g.ruledefs['r0'].rhs
        self.assertEqual(r0.lo, r0.hi)
        self.assertEqual(r0.lo.as_int(), 1)

        r1: GRep = g.ruledefs['r1'].rhs
        self.assertEqual(r1.lo, r1.hi)
        self.assertEqual(r1.lo.expr, 'x')

        r2: GRep = g.ruledefs['r2'].rhs
        self.assertEqual(r2.lo, r2.hi)
        self.assertEqual(r2.lo, None)

        r3: GRep = g.ruledefs['r3'].rhs
        self.assertEqual(r3.lo, r3.hi)
        self.assertEqual(r3.lo, None)
        self.assertIsInstance(r3.dist, RepDist)
        self.assertEqual(r3.dist.name, 'dist')
        self.assertEqual(r3.dist.args[0].as_int(), 1)

    def test_rep2(self):
        g = GrammaGrammar('''
        r0 := 'a'{2,};
        r1 := 'a'{,3};
        r2 := 'a'{4,5};
        r3 := 'a'{4,5, dist(1)};
    ''')
        for rulename, ruledef in g.ruledefs.items():
            self.assertIsInstance(ruledef.rhs, GRep)
            self.assertEqual(str(ruledef.rhs.child), "'a'")
        r: GRep
        r = g.ruledefs['r0'].rhs
        self.assertEqual(r.lo.as_int(), 2)
        self.assertEqual(r.hi, None)

        r = g.ruledefs['r1'].rhs
        self.assertEqual(r.lo, None)
        self.assertEqual(r.hi.as_int(), 3)

        r = g.ruledefs['r2'].rhs
        self.assertEqual(r.lo.as_int(), 4)
        self.assertEqual(r.hi.as_int(), 5)

        r = g.ruledefs['r3'].rhs
        self.assertEqual(r.lo.as_int(), 4)
        self.assertEqual(r.hi.as_int(), 5)
        self.assertEqual(r.dist.name, 'dist')
        self.assertEqual(r.dist.args[0].as_int(), 1)

    def test_rep3(self):
        g = GrammaGrammar('''
        r0 := 'a'{1,`x`};
        r1 := 'a'{`x`,2};
        r2 := 'a'{`x`,`y`};
        r3 := 'a'{`x`,`y`, dist(1)};
    ''')
        for rulename, ruledef in g.ruledefs.items():
            self.assertIsInstance(ruledef.rhs, GRep)
            self.assertEqual(str(ruledef.rhs.child), "'a'")
        r0: GRep = g.ruledefs['r0'].rhs
        self.assertEqual(r0.lo.as_int(), 1)
        self.assertEqual(r0.hi.expr, 'x')

        r1: GRep = g.ruledefs['r1'].rhs
        self.assertEqual(r1.lo.expr, 'x')
        self.assertEqual(r1.hi.as_int(), 2)

        r2: GRep = g.ruledefs['r2'].rhs
        self.assertEqual(r2.lo.expr, 'x')
        self.assertEqual(r2.hi.expr, 'y')

        r3: GRep = g.ruledefs['r3'].rhs
        self.assertEqual(r3.lo.expr, 'x')
        self.assertEqual(r3.hi.expr, 'y')
        self.assertEqual(r3.dist.name, 'dist')
        self.assertEqual(r3.dist.args[0].as_int(), 1)

    def test_references(self):
        """test references in choosein expressions and parameterized rules, including nesting"""
        g = GrammaGrammar('''
            r0(a,b) := a.b;
            r1 := 'c';
            r2 := choose x ~ 'd' in f0('a','b') . f1(). r0('a','b') . r1 . x;
            r3(x,y) :=  choose x ~ x in x;
        ''')
        r2 = g.ruledefs['r2'].rhs
        self.assertIsInstance(r2, GChooseIn)
        e = r2.child
        self.assertIsInstance(e.children[0], GFuncRef)
        self.assertIsInstance(e.children[1], GFuncRef)
        self.assertIsInstance(e.children[2], GRuleRef)
        self.assertIsInstance(e.children[3], GRuleRef)
        self.assertIsInstance(e.children[4], GVarRef)

        r3 = g.ruledefs['r3'].rhs
        self.assertIsInstance(r3, GChooseIn)
        self.assertIsInstance(r3.child, GVarRef)
        self.assertEqual(r3.vnames[0], 'x')
        self.assertIsInstance(r3.values[0], GVarRef)

    def test_range(self):
        g = GrammaGrammar('''
            r0 := ['a'..'b'];
            r1 := ['a'..'b', 'c'..'d'];
            r2 := ['a', 'b'..'c', 'd'];
        ''')
        for rulename, ruledef in g.ruledefs.items():
            self.assertIsInstance(ruledef.rhs, GRange)

        r: GRange
        r = cast(GRange, g.ruledefs['r0'].rhs)
        self.assertEqual(r.pairs[0][0], ord('a'))
        self.assertEqual(r.pairs[0][1], 1 + ord('b') - ord('a'))

        r = cast(GRange, g.ruledefs['r1'].rhs)
        self.assertEqual(r.pairs[0][0], ord('a'))
        self.assertEqual(r.pairs[0][1], 1 + ord('b') - ord('a'))
        self.assertEqual(r.pairs[1][0], ord('c'))
        self.assertEqual(r.pairs[1][1], 1 + ord('d') - ord('c'))

        r = cast(GRange, g.ruledefs['r2'].rhs)
        self.assertEqual(r.pairs[0][0], ord('a'))
        self.assertEqual(r.pairs[0][1], 1)
        self.assertEqual(r.pairs[1][0], ord('b'))
        self.assertEqual(r.pairs[1][1], 1 + ord('c') - ord('b'))
        self.assertEqual(r.pairs[2][0], ord('d'))
        self.assertEqual(r.pairs[2][1], 1)

    def test_denoted(self):
        g = GrammaGrammar('''
            r0 := 'a'/123;
            r1 := 'a'/'b';
            r2 := 'a'/`b`;
            r3(x) := 'a'/f(x);
        ''')
        for rulename, ruledef in g.ruledefs.items():
            self.assertIsInstance(ruledef.rhs, GDenoted)

        r = g.ruledefs['r0']
        d: GDenoted
        d = cast(GDenoted, r.rhs)
        self.assertIsInstance(d.right, GTok)
        self.assertEqual(d.right.as_int(), 123)

        r = g.ruledefs['r1']
        d = cast(GDenoted, r.rhs)
        self.assertIsInstance(d.right, GTok)
        self.assertEqual(d.right.as_str(), 'b')

        r = g.ruledefs['r2']
        d = cast(GDenoted, r.rhs)
        self.assertIsInstance(d.right, GCode)
        self.assertEqual(d.right.expr, 'b')

        r = g.ruledefs['r3']
        d = cast(GDenoted, r.rhs)
        self.assertIsInstance(d.right, GDFuncRef)
        self.assertEqual(d.right.fname, 'f')
        self.assertIsInstance(d.right.fargs[0], GVarRef)


class TestCopyStrSimplify(unittest.TestCase):
    def test_cat_alt(self):
        g = GrammaGrammar('''
            start := ('a'|'b').('c'|'d');
        ''')

        ge = cast(GCat, g.ruledefs['start'].rhs)
        self.assertIsInstance(ge, GCat)
        self.assertIsInstance(ge.children[0], GAlt)
        self.assertEqual(str(ge), "('a'|'b').('c'|'d')")

        ge2 = cast(GCat, ge.copy())
        self.assertIsInstance(ge2, GCat)
        self.assertIsInstance(ge2.children[0], GAlt)
        self.assertEqual(str(ge2), str(ge))

        ge3 = cast(GCat, ge.simplify())
        self.assertIsInstance(ge3, GCat)
        self.assertIsInstance(ge3.children[0], GAlt)
        self.assertEqual(str(ge2), str(ge))

    def test_rep_range(self):
        g = GrammaGrammar('''
            start := ['a'..'b']{1,3};
        ''')

        ge = cast(GRep, g.ruledefs['start'].rhs)
        self.assertIsInstance(ge, GRep)
        self.assertIsInstance(ge.children[0], GRange)
        self.assertEqual(str(ge), "['a'..'b']{1,3}")

        ge2 = cast(GRep, ge.copy())
        self.assertIsInstance(ge2, GRep)
        self.assertIsInstance(ge2.children[0], GRange)
        self.assertEqual(str(ge2), str(ge))

        ge3 = cast(GRep, ge.simplify())
        self.assertIsInstance(ge3, GRep)
        self.assertIsInstance(ge3.children[0], GRange)
        self.assertEqual(str(ge3), str(ge))

    def test_choosein_tern(self):
        g = GrammaGrammar('''
            start := choose x~'a' in `1`?x:'d';
        ''')

        ge = cast(GChooseIn, g.ruledefs['start'].rhs)
        self.assertIsInstance(ge, GChooseIn)
        self.assertIsInstance(ge.child, GTern)
        self.assertEqual(str(ge), "choose x~'a' in `1`?x:'d'")

        ge2 = cast(GChooseIn, ge.copy())
        self.assertIsInstance(ge2, GChooseIn)
        self.assertIsInstance(ge2.child, GTern)
        self.assertEqual(str(ge2), str(ge))

        ge3 = cast(GChooseIn, ge.simplify())
        self.assertIsInstance(ge3, GChooseIn)
        self.assertIsInstance(ge3.child, GTern)
        self.assertEqual(str(ge3), str(ge))

    def test_rule_denoted(self):
        g = GrammaGrammar('''
            start := r('a','b');
            r(x,y) := x/`1`;
        ''')

        ge = cast(GRuleRef, g.ruledefs['start'].rhs)
        self.assertIsInstance(ge, GRuleRef)
        self.assertEqual(str(ge), "r('a','b')")

        ge2 = cast(GRuleRef, ge.copy())
        self.assertIsInstance(ge2, GRuleRef)
        self.assertEqual(str(ge2), str(ge))

        ge3 = cast(GRuleRef, ge.simplify())
        self.assertIsInstance(ge3, GRuleRef)
        self.assertEqual(str(ge3), str(ge))

        ge = cast(GRuleRef, g.ruledefs['r'].rhs)
        self.assertIsInstance(ge, GDenoted)
        self.assertEqual(str(ge), "x/`1`")

        ge2 = cast(GRuleRef, ge.copy())
        self.assertIsInstance(ge2, GDenoted)
        self.assertEqual(str(ge2), str(ge))

        ge3 = cast(GRuleRef, ge.simplify())
        self.assertIsInstance(ge3, GDenoted)
        self.assertEqual(str(ge3), str(ge))

    def test_reps(self):
        g = GrammaGrammar('''
            r0 := 'a'{,};
            r1 := 'a'{1};
            r2 := 'a'{1,2};
            r3 := 'a'{,2};
            r4 := 'a'{1,};
            r5 := 'a'{1,2,geom(5)};
            r6 := 'a'{geom(5)};
        ''')
        r: GRep

        r = g.ruledefs['r0'].rhs
        self.assertEqual(str(r), "'a'{,}")
        self.assertEqual(str(r.copy()), "'a'{,}")
        self.assertEqual(str(r.simplify()), "'a'{,}")

        r = g.ruledefs['r1'].rhs
        self.assertEqual(str(r), "'a'{1}")
        self.assertEqual(str(r.copy()), "'a'{1}")
        self.assertEqual(str(r.simplify()), "'a'{1}")

        r = g.ruledefs['r2'].rhs
        self.assertEqual(str(r), "'a'{1,2}")
        self.assertEqual(str(r.copy()), "'a'{1,2}")
        self.assertEqual(str(r.simplify()), "'a'{1,2}")

        r = g.ruledefs['r3'].rhs
        self.assertEqual(str(r), "'a'{,2}")
        self.assertEqual(str(r.copy()), "'a'{,2}")
        self.assertEqual(str(r.simplify()), "'a'{,2}")

        r = g.ruledefs['r4'].rhs
        self.assertEqual(str(r), "'a'{1,}")
        self.assertEqual(str(r.copy()), "'a'{1,}")
        self.assertEqual(str(r.simplify()), "'a'{1,}")

        r = g.ruledefs['r5'].rhs
        self.assertEqual(str(r), "'a'{1,2,geom(5)}")
        self.assertEqual(str(r.copy()), "'a'{1,2,geom(5)}")
        self.assertEqual(str(r.simplify()), "'a'{1,2,geom(5)}")

        r = g.ruledefs['r6'].rhs
        self.assertEqual(str(r), "'a'{geom(5)}")
        self.assertEqual(str(r.copy()), "'a'{geom(5)}")
        self.assertEqual(str(r.simplify()), "'a'{geom(5)}")

    def test_alts(self):
        g = GrammaGrammar('''
            r1 := 'a'|'b';
            r2 := 1 'a'|'b';
            r3 := 2 'a'|'b';
            r4 := 2.3 'a'|'b';
            r5 := `1` 'a'|'b';
            r6 := ('a' | 'b') | 'c';
            r7 := 'a' | 'a' | 'b';
            r8 := 'a' | 'a';
            r9 := 'a' | 0 'b';
            r10 := 0 'a' | 0 'b';
        ''')
        r: GAlt

        r = g.ruledefs['r1'].rhs
        self.assertEqual(str(r), "'a'|'b'")
        self.assertEqual(str(r.copy()), "'a'|'b'")
        self.assertEqual(str(r.simplify()), "'a'|'b'")

        r = g.ruledefs['r2'].rhs
        self.assertEqual(str(r), "'a'|'b'")
        self.assertEqual(str(r.copy()), "'a'|'b'")
        self.assertEqual(str(r.simplify()), "'a'|'b'")

        r = g.ruledefs['r3'].rhs
        self.assertEqual(str(r), "2 'a'|'b'")
        self.assertEqual(str(r.copy()), "2 'a'|'b'")
        self.assertEqual(str(r.simplify()), "2 'a'|'b'")

        r = g.ruledefs['r4'].rhs
        self.assertEqual(str(r), "2.3 'a'|'b'")
        self.assertEqual(str(r.simplify()), "2.3 'a'|'b'")
        self.assertEqual(str(r.copy()), "2.3 'a'|'b'")

        r = g.ruledefs['r5'].rhs
        self.assertEqual(str(r), "`1` 'a'|'b'")
        self.assertEqual(str(r.copy()), "`1` 'a'|'b'")
        self.assertEqual(str(r.simplify()), "`1` 'a'|'b'")

        r = g.ruledefs['r6'].rhs
        self.assertEqual(str(r), "('a'|'b')|'c'")
        self.assertEqual(str(r.copy()), "('a'|'b')|'c'")
        self.assertEqual(str(r.simplify()), "'a'|'b'|2 'c'")

        r = g.ruledefs['r7'].rhs
        self.assertEqual(str(r), "'a'|'a'|'b'")
        self.assertEqual(str(r.copy()), "'a'|'a'|'b'")
        self.assertEqual(str(r.simplify()), "2 'a'|'b'")

        r = g.ruledefs['r8'].rhs
        self.assertEqual(str(r), "'a'|'a'")
        self.assertEqual(str(r.copy()), "'a'|'a'")
        self.assertEqual(str(r.simplify()), "'a'")

        r = g.ruledefs['r9'].rhs
        self.assertEqual(str(r), "'a'|0 'b'")
        self.assertEqual(str(r.copy()), "'a'|0 'b'")
        self.assertEqual(str(r.simplify()), "'a'")

        r = g.ruledefs['r10'].rhs
        self.assertEqual(str(r), "0 'a'|0 'b'")
        self.assertEqual(str(r.copy()), "0 'a'|0 'b'")
        self.assertEqual(str(r.simplify()), "''")

    def test_cats(self):
        g = GrammaGrammar('''
            r1 := ('a'.'b').'c';
            r2 := (['a'..'c'].['d'..'f']).['g'..'i'];
        ''')
        r: GCat
        r = g.ruledefs['r1'].rhs
        self.assertEqual(str(r), "'a'.'b'.'c'")
        self.assertEqual(str(r.copy()), "'a'.'b'.'c'")
        self.assertEqual(str(r.simplify()), "'abc'")

        r = g.ruledefs['r2'].rhs
        self.assertEqual(str(r), "['a'..'c'].['d'..'f'].['g'..'i']")
        self.assertEqual(str(r.copy()), "['a'..'c'].['d'..'f'].['g'..'i']")
        self.assertEqual(str(r.simplify()), "['a'..'c'].['d'..'f'].['g'..'i']")

        r = r.simplify()
        del r.children[1:]
        self.assertEqual(str(r), "['a'..'c']")
        self.assertEqual(str(r.simplify()), "['a'..'c']")
        del r.children[:]
        self.assertEqual(str(r), "''")
        self.assertEqual(str(r.simplify()), "''")

    def test_ranges(self):
        g = GrammaGrammar('''
            r1 := ['a'..'b'];
            r2 := ['a'];
            r3 := ['a','b','c'];
            r4 := ['a'..'c','d'..'f'];
            r5 := ['a'..'b','d'..'e'];
        ''')
        r: GRange
        r = g.ruledefs['r1'].rhs
        self.assertEqual(str(r), "['a'..'b']")
        self.assertEqual(str(r.copy()), "['a'..'b']")
        self.assertEqual(str(r.simplify()), "['a'..'b']")

        r = g.ruledefs['r2'].rhs
        self.assertEqual(str(r), "['a']")
        self.assertEqual(str(r.copy()), "['a']")
        self.assertEqual(str(r.simplify()), "'a'")

        r = g.ruledefs['r3'].rhs
        self.assertEqual(str(r), "['a','b','c']")
        self.assertEqual(str(r.copy()), "['a','b','c']")
        self.assertEqual(str(r.simplify()), "['a'..'c']")

        r = g.ruledefs['r4'].rhs
        self.assertEqual(str(r), "['a'..'c','d'..'f']")
        self.assertEqual(str(r.copy()), "['a'..'c','d'..'f']")
        self.assertEqual(str(r.simplify()), "['a'..'f']")

        r = g.ruledefs['r5'].rhs
        self.assertEqual(str(r), "['a'..'b','d'..'e']")
        self.assertEqual(str(r.copy()), "['a'..'b','d'..'e']")
        self.assertEqual(str(r.simplify()), "['a'..'b','d'..'e']")

    def test_nesting(self):
        g = GrammaGrammar('''
            r1:= ('a'/2).'b';
            r2:= ('a'|'b'){3};
        ''')
        r: GExpr
        r = g.ruledefs['r1'].rhs
        self.assertEqual(str(r), "('a'/2).'b'")

        r = g.ruledefs['r2'].rhs
        self.assertEqual(str(r), "('a'|'b'){3}")

    def test_terns(self):
        g = GrammaGrammar('''
            r1:=`1`?'a':'b';
        ''')
        r = g.ruledefs['r1'].rhs
        self.assertEqual(str(r), "`1`?'a':'b'")
        self.assertEqual(str(r.copy()), "`1`?'a':'b'")
        self.assertEqual(str(r.simplify()), "`1`?'a':'b'")

    def test_refs(self):
        g = GrammaGrammar('''
            r1:= r.f(r).r/df(`x`);
            r := 'a';
        ''')
        r = g.ruledefs['r1'].rhs
        # from IPython import embed;embed()
        self.assertEqual(str(r), "r.f(r).r/df(`x`)")
        self.assertEqual(str(r.copy()), "r.f(r).r/df(`x`)")
        self.assertEqual(str(r.simplify()), "r.f(r).r/df(`x`)")


class TestNavigation(unittest.TestCase):
    def test_isruleref(self):
        g = GrammaGrammar('''
           start := r.('a'|'b').['a'..'b'].(choose x~'a' in x.x);
           r:='b';
        ''')
        children = g.ruledefs['start'].rhs.children
        self.assertTrue(children[0].is_ruleref())
        self.assertTrue(children[0].is_ruleref('r'))
        self.assertFalse(children[0].is_ruleref('s'))
        self.assertFalse(children[1].is_ruleref())
        self.assertFalse(children[2].is_ruleref())
        self.assertFalse(children[3].is_ruleref())

    def test_ancestry(self):
        g = GrammaGrammar('''
           r1 := choose x~'a' in (x.'a'|'b'){1,3};
        ''')

        ge1 = g.ruledefs['r1'].rhs
        self.assertIsInstance(ge1, GChooseIn)
        ge2 = ge1.child
        self.assertIsInstance(ge2, GRep)
        ge3 = ge2.child
        self.assertIsInstance(ge3, GAlt)
        ge4 = ge3.children[0]
        self.assertIsInstance(ge4, GCat)
        ge5 = ge4.children[0]
        self.assertIsInstance(ge5, GVarRef)

        self.assertIs(ge5.get_ancestor(GCat), ge4)
        self.assertIs(ge5.get_ancestor(GAlt), ge3)
        self.assertIs(ge5.get_ancestor(GRep), ge2)
        self.assertIs(ge5.get_ancestor(GChooseIn), ge1)

        self.assertIsNone(ge5.get_ancestor(GTok))

    def test_walk(self):
        g = GrammaGrammar('''
           r1 := choose x~'a' in (x.'a'|'b'){1,3};
        ''')

        ge = g.ruledefs['r1'].rhs
        gel = list(ge.walk())
        self.assertEqual(len(gel), 8)

        self.assertEqual(','.join(ge.__class__.__name__ for ge in gel),
                         'GChooseIn,GTok,GRep,GAlt,GCat,GVarRef,GTok,GTok')


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3

import unittest

from gramma.parser import GrammaGrammar, GFuncRef, GRuleRef, GChooseIn, GVarRef, GAlt, GRep, RepDist, GRange, GDenoted, \
    GTok, GCode, GDFuncRef

# from IPython import embed;embed()


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


class GDFunc(object):
    pass


class TestGrammaGrammar(unittest.TestCase):
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
        self.assertEqual(str(g.ruledefs['r3'].rhs), """`True` ? 'a' : 'b'""")
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
        r0: GRep = g.ruledefs['r0'].rhs
        self.assertEqual(r0.lo.as_int(), 2)
        self.assertEqual(r0.hi, None)

        r1: GRep = g.ruledefs['r1'].rhs
        self.assertEqual(r1.lo, None)
        self.assertEqual(r1.hi.as_int(), 3)

        r2: GRep = g.ruledefs['r2'].rhs
        self.assertEqual(r2.lo.as_int(), 4)
        self.assertEqual(r2.hi.as_int(), 5)

        r3: GRep = g.ruledefs['r3'].rhs
        self.assertEqual(r3.lo.as_int(), 4)
        self.assertEqual(r3.hi.as_int(), 5)
        self.assertEqual(r3.dist.name, 'dist')
        self.assertEqual(r3.dist.args[0].as_int(), 1)

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

        r0: GRange = g.ruledefs['r0'].rhs
        self.assertEqual(r0.pairs[0][0], ord('a'))
        self.assertEqual(r0.pairs[0][1], 1 + ord('b') - ord('a'))

        r1: GRange = g.ruledefs['r1'].rhs
        self.assertEqual(r1.pairs[0][0], ord('a'))
        self.assertEqual(r1.pairs[0][1], 1 + ord('b') - ord('a'))
        self.assertEqual(r1.pairs[1][0], ord('c'))
        self.assertEqual(r1.pairs[1][1], 1 + ord('d') - ord('c'))

        r2: GRange = g.ruledefs['r2'].rhs
        self.assertEqual(r2.pairs[0][0], ord('a'))
        self.assertEqual(r2.pairs[0][1], 1)
        self.assertEqual(r2.pairs[1][0], ord('b'))
        self.assertEqual(r2.pairs[1][1], 1 + ord('c') - ord('b'))
        self.assertEqual(r2.pairs[2][0], ord('d'))
        self.assertEqual(r2.pairs[2][1], 1)

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
        self.assertIsInstance(r.rhs.right, GTok)
        self.assertEqual(r.rhs.right.as_int(), 123)

        r = g.ruledefs['r1']
        self.assertIsInstance(r.rhs.right, GTok)
        self.assertEqual(r.rhs.right.as_str(), 'b')

        r = g.ruledefs['r2']
        self.assertIsInstance(r.rhs.right, GCode)
        self.assertEqual(r.rhs.right.expr, 'b')

        r = g.ruledefs['r3']
        self.assertIsInstance(r.rhs.right, GDFuncRef)
        self.assertEqual(r.rhs.right.fname, 'f')
        self.assertIsInstance(r.rhs.right.fargs[0], GVarRef)


if __name__ == '__main__':
    unittest.main()
#!/usr/bin/env python3

import unittest
from gramma.parser import GrammaGrammar, GCat, GFunc, GRule, GChooseIn, GVar

smt_glf = '''\
start := equals(sort);
equals(s) := '(= '.sexpr(s).' '.sexpr(s).' )';

# sexprs
sexpr(s) := switch_sort(s, int_sexpr, bool_sexpr, array_sexpr(domain(s), range(s)));

bool_sexpr := 'true' | 'false'
            | array_wrap(bool_sort)
            ;
int_sexpr := ['1'..'9'].['0'..'9']{geom(4)}
            | array_wrap(int_sort)
            ;
array_sexpr(domain_sort, range_sort) :=  
              "(as const (Array ".domain_sort." ".range_sort."))"
            | array_variable(domain_sort, range_sort)
            | "(store ".array_sexpr(domain_sort, range_sort)." ".sexpr(domain_sort)." ".sexpr(range_sort).")"
            ;
const_array_sexpr :=
            '(store (store (store ((as const (Array Int Int)) 0) 0 1) 1 2) 0 0)';
# sorts
sort := int_sort | bool_sort | array_sort(sort,sort);
int_sort := 'Int'/'i';
bool_sort := 'Bool'/'b';
array_sort(d, r) := '( Array '.d.' '.r.' )' / make_array(d,r);

# generate an sexpr of the given type by invoking an array whose range is the given type
array_wrap(s) := choose domain~sort in 
            "( select " . array_sexpr(array_sort(domain, s))." ".sexpr(domain).")";
'''

GrammaGrammar.GLF_PARSER.parse(smt_glf)

# from IPython import embed;embed()


class TestLarkParser(unittest.TestCase):
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
        self.assertEqual(args.children[3].data, 'number')
        self.assertEqual(args.children[3].children[0].type, 'FLOAT')
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
        ''')
        self.assertTrue(all(c.data == 'ruledef' for c in lt.children), msg='all children should be ruledefs')
        self.assertEqual(len(lt.children), 10, msg='wrong number of ruledefs')
        self.assertEqual(lt.children[0].children[1].data, 'string')
        self.assertEqual(lt.children[1].children[1].data, 'choosein')
        self.assertEqual(lt.children[2].children[1].data, 'alt')
        self.assertEqual(lt.children[3].children[1].data, 'tern')
        self.assertEqual(lt.children[4].children[1].data, 'den')
        self.assertEqual(lt.children[5].children[1].data, 'cat')
        self.assertEqual(lt.children[6].children[1].data, 'rep')
        self.assertEqual(lt.children[7].children[1].data, 'range')
        self.assertEqual(lt.children[8].children[1].data, 'func')

        self.assertTrue(all(len(c.children) == 2 for c in lt.children[:9]),
                        msg='non parameterized ruledefs should have 2 children')
        self.assertEqual(len(lt.children[9].children), 3,
                         msg='parameterized ruledefs should have 3 children')

        self.assertEqual(lt.children[9].children[1].data, 'rule_parms')
        self.assertEqual(lt.children[9].children[2].data, 'den')


class TestGrammaGrammar(unittest.TestCase):
    def test_sampler(self):
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
        self.assertEqual(str(g.ruledefs['r0'][0]), """'a'""")
        self.assertEqual(str(g.ruledefs['r1'][0]), """choose x~'a' in 'b'""")
        self.assertEqual(str(g.ruledefs['r2'][0]), """'a'|'b'""")
        self.assertEqual(str(g.ruledefs['r3'][0]), """`True` ? 'a' : 'b'""")
        self.assertEqual(str(g.ruledefs['r4'][0]), """'a'/'b'""")
        self.assertEqual(str(g.ruledefs['r5'][0]), """'a'.'b'""")
        self.assertEqual(str(g.ruledefs['r6'][0]), """'a'{1,3}""")
        self.assertEqual(str(g.ruledefs['r7'][0]), """['a'..'b']""")
        self.assertEqual(str(g.ruledefs['r8'][0]), """f('a',`x`,10,1.2)""")
        self.assertEqual(str(g.ruledefs['r9'][0]), """a/'b'""")

    def test_references(self):
        g = GrammaGrammar('''
            r0(a,b) := a.b;
            r1 := 'c';
            r2 := choose x ~ 'd' in f0('a','b') . f1. r0('a','b') . r1 . x;
        ''')
        r2 = g.ruledefs['r2'][0]
        self.assertIsInstance(r2, GChooseIn)
        e = r2.child
        self.assertIsInstance(e.children[0], GFunc)
        self.assertIsInstance(e.children[1], GFunc)
        self.assertIsInstance(e.children[2], GRule)
        self.assertIsInstance(e.children[3], GRule)
        self.assertIsInstance(e.children[4], GVar)


if __name__ == '__main__':
    unittest.main()

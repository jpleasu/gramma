#!/usr/bin/env python3
'''
- emits "neg" gfunc for negated character ranges.
- emits "`maxrep`" upper range value for unbounded iterations

TODO
====
- generator should insert whitespace between tokens if g4 ignores it.. each
  (non fragment?) token could be preceded by a space?
'''
from __future__ import absolute_import, division, print_function

import os
import sys

import antlr4
from antlr4.tree.Tree import TerminalNode

sys.path.append(os.path.join(os.path.dirname(__file__), 'antlr4parser'))

from ANTLRv4LexerPythonTarget import ANTLRv4LexerPythonTarget as ANTLRv4Lexer
from ANTLRv4Parser import ANTLRv4Parser


def dump(n, indent='', file=sys.stdout):
    if n.getChildCount() > 0:
        print('%s%s' % (indent, type(n).__name__), file=file)
        for c in n.getChildren():
            dump(c, indent + '  ', file=file)
    elif isinstance(n, TerminalNode):
        print('%s"%s"' % (indent, n.getText()), file=file)
    else:
        print('%s%s: "%s"' % (indent, type(n).__name__, n.getText()), file=file)


def gchar(c):
    'unicode char -> gramma char'
    u8 = c.encode('utf8')
    if len(u8) == 1:
        return repr(u8)[1:]  # drop the b in b'x'
    return repr(c)


class TransformingVisitor(antlr4.ParseTreeVisitor):
    def __init__(self, g4filename):
        self.g4filename = g4filename
        self.rules = []

    def defaultResult(self):
        return None

    def visitGrammarSpec(self, gs):
        return ''.join(self.visit(r) for r in gs.rules().ruleSpec())

    def visitChildren(self, node):
        # "skip" nodes with one child
        if node.getChildCount() == 1:
            return self.visit(node.getChild(0))
        print('==== unhandled Antlr node \"%s\" ====' % type(node).__name__, file=sys.stderr)
        dump(node, file=sys.stderr)
        print('----------------------------', file=sys.stderr)
        from IPython import embed;
        embed()
        return 'g4toglf_fail_%s(%s)' % (type(node).__name__, ','.join(self.visit(c) for c in node.getChildren()))

    def visitElement(self, ec):
        if isinstance(ec.getChild(0), ANTLRv4Parser.ActionBlockContext):
            return '''action(%s)''' % repr(ec.getText())
        return ''.join(self.visit(c) for c in ec.getChildren())

    visitLexerElement = visitElement

    def visitElementOptions(self, eo):
        return ''  # e.g. <assoc=right>

    def aggregateResult(self, result, childResult):
        return result + childResult

    def do_rule(self, lhs, rhs):
        rulename = lhs.getText()
        self.rules.append(rulename)
        return ('%s := %s ;\n' % (rulename, self.visit(rhs)))

    def visitRuleAltList(self, alt):
        g = (self.visit(c) for c in alt.getChildren() if not isinstance(c, TerminalNode))
        return '|'.join(s for s in g if s != '')

    def visitAlternative(self, alt):
        if alt.getChildCount() == 0:
            return "''"
        s = '.'.join(self.visit(e) for e in alt.element())
        if alt.elementOptions() != None:
            toRep = self.visit(alt.elementOptions())
            if toRep == '':
                return ''
            if s == '':
                return '(%s){0,1}' % (toRep)
            return '(%s){0,1}.%s' % (toRep, s)
        return s

    def visitCharacterRange(self, r):
        return '[' + self.do_range(r) + ']'

    def do_range(self, r):
        s = r.getText()
        if s.startswith('~'):
            s = s[1:]
            neg = True
        else:
            neg = False
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
            s = eval("u'''%s'''" % s)
            ns = []
            i = 0
            if s[0] == '-':
                ns.append(gchar(s[i]))
                i += 1
            while i < len(s) - 1:
                if s[i + 1] == '-' and i < len(s) - 2:
                    ns.append('[%s..%s]' % (gchar(s[i]), gchar(s[i + 2])))
                    i += 3
                else:
                    ns.append(gchar(s[i]))
                    i += 1
            if i < len(s):
                ns.append(gchar(s[i]))

            if len(ns) > 1:
                ns = '(%s)' % '|'.join(ns)
            else:
                ns = ns[0]
            if neg:
                return 'neg(%s)' % ns
            return ns
        elif s[0] == "'" and s[-1] == "'":
            # single character
            return s
        else:
            return 'PARSEME_RANGE(%s)' % r.getText()

    def visitNotSet(self, ns):
        return 'neg(%s)' % self.visit(ns.getChild(1))

    def visitBlockSet(self, e):
        children = list(e.getChildren())[1:-1]
        # block sets are only within NotSet, so don't wrap w/ parens
        return ''.join(self.visit(c) for c in children)  # use existing '|' tokens from antlr

    def visitSetElement(self, e):
        return self.do_range(e)

    def visitTerminal(self, la):
        s = la.getText()
        if s.startswith("'") and s.endswith("'"):
            # return '"%s"' % s[1:-1]
            return s  # o/w need to escape quotes
        elif s.startswith("[") and s.endswith("]"):
            return self.do_range(la)
        elif s == '.':
            return 'any()'
        elif s == 'EOF':
            return 'eof()'
        return s

    def visitEbnf(self, ebnf):
        return ''.join(self.visit(c) for c in ebnf.getChildren())

    def visitEbnfSuffix(self, s):
        s = s.getText()
        ## suffix can be lazy, e.g. "*?".. we don't care.
        if s[0] == '*':
            return '{0,`maxrep`}'
        elif s[0] == '+':
            return '{1,`maxrep`}'
        elif s[0] == '?':
            return '{0,1}'
        else:
            return 'PARSEME_SUFFIX(%s)' % s

    def visitParserRuleSpec(self, pr):
        lhs = pr.RULE_REF()
        rhs = pr.ruleBlock()
        return self.do_rule(lhs, rhs)

    def visitLexerBlock(self, lb):
        return '(%s)' % (self.visit(lb.lexerAltList()))

    def visitBlock(self, b):
        return '(%s)' % (self.visit(b.altList()))

    def visitLexerRuleSpec(self, lr):
        lhs = lr.TOKEN_REF()
        rhs = lr.lexerRuleBlock()
        return self.do_rule(lhs, rhs)

    def visitLexerAltList(self, alt):
        # return '%d(%s)' % (alt.getChildCount(), '|'.join(self.visit(c) for c in alt.getChildren()))
        # return '|'.join('[%s]'% self.visit(c) for c in alt.getChildren() if not isinstance(c, TerminalNode))
        return '|'.join(self.visit(c) for c in alt.getChildren() if not isinstance(c, TerminalNode))

    visitAltList = visitLexerAltList

    def visitLexerAlt(self, le):
        # XXX ignoring le.lexerCommands
        elements = le.lexerElements()
        if elements == None:
            return "''"  # e.g. the first part of BLAH : (|'a')
        else:
            return '.'.join(self.visit(e) for e in elements.lexerElement())

    def visitLabeledAlt(self, la):
        return self.visit(la.getChild(0))  # drop the label in  "blah # Label" -> "blah"

    def parseAndWrite(self, out, final=False):
        inp = antlr4.FileStream(self.g4filename, encoding='utf8')
        lexer = ANTLRv4Lexer(inp)
        stream = antlr4.CommonTokenStream(lexer)
        parser = ANTLRv4Parser(stream)
        tree = parser.grammarSpec()
        # print('file: %s' % self.g4filename, file=sys.stderr)
        # dump(tree, file=sys.stderr)
        s = self.visit(tree)
        if final and not 'start' in self.rules:
            s += '\nstart:=%s;\n' % self.rules[0]
        out.write(s)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='generate gramma GLF from ANLTR4 (.g4) grammar')
    parser.add_argument('g4files', metavar='G4_IN',
                        type=argparse.FileType('r'), nargs='+', help='input ANTLR4 G4 file(s)')
    parser.add_argument('-o', '--out', dest='glf', metavar='GLF_OUT',
                        type=argparse.FileType('w'), help='output GLF file', default='-')
    args = parser.parse_args()

    for g4file in args.g4files[:-1]:
        TransformingVisitor(g4file.name).parseAndWrite(args.glf)
    TransformingVisitor(args.g4files[-1].name).parseAndWrite(args.glf, final=True)

# vim: ts=4 sw=4

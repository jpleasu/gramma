#!/usr/bin/env python2
'''
- emits "neg" gfunc for negated character ranges.
- emits "`maxrep`" upper range value for unbounded iterations

TODO
====
- generator should insert whitespace between tokens if g4 ignores it.. each
  (non fragment?) token could be preceded by a space?
'''
from __future__ import absolute_import, division, print_function

import sys

import antlr4
from antlr4.tree.Tree import RuleNode, ErrorNode, TerminalNode

from antlr4parser.ANTLRv4LexerPythonTarget import ANTLRv4LexerPythonTarget as ANTLRv4Lexer
from antlr4parser.ANTLRv4Parser import ANTLRv4Parser 
#from antlr4parser.ANTLRv4ParserVisitor import ANTLRv4ParserVisitor

from builtins import super

def dump(n,indent='',file=sys.stdout):
  if n.getChildCount()>0:
    print('%s%s' % (indent,type(n).__name__),file=file)
    for c in n.getChildren():
      dump(c,indent+'  ',file=file)
  elif isinstance(n, TerminalNode):
    print('%s"%s"' % (indent,n.getText()),file=file)
  else:
    print('%s%s: "%s"' % (indent,type(n).__name__, n.getText()),file=file)

def gchar(c):
  'unicode char -> gramma char'
  u8=c.encode('utf8')
  if len(u8)==1:
    return repr(u8)
  return repr(c)

class TransformingVisitor(antlr4.ParseTreeVisitor):
  def defaultResult(self):
    return None

  def visitGrammarSpec(self,gs):
    return ''.join(self.visit(r) for r in gs.rules().ruleSpec())

  def visitChildren(self, node):
    if node.getChildCount()==1:
      return self.visit(node.getChild(0))
    return '%s(%s)' % (type(node).__name__, ','.join(self.visit(c) for c in node.getChildren()))

  def visitElement(self,ec):
    return ''.join(self.visit(c) for c in ec.getChildren())
  visitLexerElement=visitElement


  def aggregateResult(self,result,childResult):
    return result + childResult

  def do_rule(self,lhs,rhs):
    return ('%s := %s ;\n' % (lhs.getText(), self.visit(rhs)))

  def visitRuleAltList(self,alt):
    return '|'.join(self.visit(c) for c in alt.getChildren() if not isinstance(c, TerminalNode))
    
  def visitAlternative(self, alt):
    if alt.getChildCount()==0:
      return "''"
    s='.'.join(self.visit(e) for e in alt.element())
    if alt.elementOptions()!=None:
      if s=='':
        return '(%s){0,1}' % (self.visit(alt.elementOptions()))
      return '(%s){0,1}.%s' % (self.visit(alt.elementOptions()),s)
    return s

  def do_range(self,r):
    s=r.getText()
    if s.startswith('~'):
      s=s[1:]
      neg=True
    else:
      neg=False
    if s.startswith('[') and s.endswith(']'):
      s=s[1:-1]
      s=eval("u'''%s'''"%s)
      ns=[]
      i=0
      if s[0]=='-':
        ns.append(gchar(s[i]))
        i+=1
      while i<len(s)-1:
        if s[i+1]=='-' and i<len(s)-2:
          ns.append('[%s..%s]' % (gchar(s[i]),gchar(s[i+2])))
          i+=3
        else:
          ns.append(gchar(s[i]))
          i+=1
      if i<len(s):
        ns.append(gchar(s[i]))

      if len(ns)>1:
        ns = '(%s)' % '|'.join(ns)
      else:
        ns = ns[0]
      if neg:
        return 'neg(%s)' % ns
      return ns
    elif s[0]=="'" and s[-1]=="'":
      # single character
      return s
    else:
      return 'PARSEME_RANGE(%s)' % r.getText()

  def visitNotSet(self,ns):
    return 'neg(%s)' % self.visit(ns.getChild(1))
  def visitBlockSet(self,e):
    children=list(e.getChildren())[1:-1]
    # block sets are only within NotSet, so don't wrap w/ parens
    return ''.join(self.visit(c) for c in children) # use existing '|' tokens from antlr

  def visitSetElement(self,e):
    return self.do_range(e)

  def visitTerminal(self, la):
    s=la.getText()
    if s.startswith("'") and s.endswith("'"):
      #return '"%s"' % s[1:-1]
      return s # o/w need to escape quotes
    elif s.startswith("[") and s.endswith("]"):
      return self.do_range(la)
    elif s=='.':
      return 'any()'
    elif s=='EOF':
      return 'eof()'
    return s


  def visitEbnf(self,ebnf):
    return ''.join(self.visit(c) for c in ebnf.getChildren())

  def visitEbnfSuffix(self,s):
    s=s.getText()
    ## suffix can be lazy, e.g. "*?".. we don't care.
    if s[0]=='*':
      return '{0,,`maxrep`}'
    elif s[0]=='+':
      return '{1,,`maxrep`}'
    elif s[0]=='?':
      return '{0,1}'
    else:
      return 'PARSEME_SUFFIX(%s)' % s

  def visitParserRuleSpec(self, pr):
    lhs=pr.RULE_REF()
    rhs=pr.ruleBlock()
    return self.do_rule(lhs,rhs)


  def visitLexerBlock(self,lb):
    return '(%s)' % (self.visit(lb.lexerAltList()))
  def visitBlock(self,b):
    return '(%s)' % (self.visit(b.altList()))


  def visitLexerRuleSpec(self, lr):
    lhs=lr.TOKEN_REF()
    rhs=lr.lexerRuleBlock()
    return self.do_rule(lhs,rhs)

  def visitLexerAltList(self, alt):
    #return '%d(%s)' % (alt.getChildCount(), '|'.join(self.visit(c) for c in alt.getChildren()))
    #return '|'.join('[%s]'% self.visit(c) for c in alt.getChildren() if not isinstance(c, TerminalNode))
    return '|'.join(self.visit(c) for c in alt.getChildren() if not isinstance(c, TerminalNode))
  visitAltList=visitLexerAltList

  def visitLexerAlt(self, le):
    # XXX ignoring le.lexerCommands
    return '.'.join(self.visit(e) for e in le.lexerElements().lexerElement())

def gettree():
  #lexer = ANTLRv4Lexer(antlr4.FileStream('grammars-v4/antlr4/examples/Hello.g4'))
  lexer = ANTLRv4Lexer(antlr4.FileStream('grammars-v4/antlr4/examples/CPP14.g4', encoding='utf8'))
  stream = antlr4.CommonTokenStream(lexer)
  parser = ANTLRv4Parser(stream)
  return parser.grammarSpec()

def do_dump():
  tree=gettree()
  dump(tree)

def do_visit():
  tree=gettree()
  v=TransformingVisitor()
  print(v.visit(tree))


if __name__ == '__main__':
  #do_dump()
  #do_visit()

  lexer = ANTLRv4Lexer(antlr4.FileStream(sys.argv[1],encoding='utf8'))
  stream = antlr4.CommonTokenStream(lexer)
  parser = ANTLRv4Parser(stream)
  tree = parser.grammarSpec()
  dump(tree)
  print(TransformingVisitor().visit(tree),file=sys.stderr)
  #print(TransformingVisitor().visit(tree))

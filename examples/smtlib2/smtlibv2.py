#!/usr/bin/env python3

from gramma import *

from collections import Counter


class RuleVisitDumper(SideEffect):
  def __init__(self):
    self.count=Counter()
  def push(self,x,ge):
    if isinstance(ge,GRule):
      self.count[ge.rname]+=1
      #print(self.count.most_common(5))
      print(self.count.most_common())
    return False

class StackWatcher(SideEffect):
  def reset_state(self,state):
    state.stk=[]
  
  def push(self,x,ge):
    if isinstance(ge,GRule):
      x.state.stk.append(ge.rname)
      c=Counter(x.state.stk)
      print(c.most_common(5))
      return True
    return False
  
  def pop(self,x,w,s):
    if w:
      del x.state.stk[-1]

class SMTLIBv2(GrammaGrammar):
  def __init__(self):
    with open('SMTLIBv2.glf') as infile:
      #GrammaGrammar.__init__(self,infile.read(), param_ids='maxrep sortrec exprrec termrec'.split(), sideeffects=[StackWatcher()])
      GrammaGrammar.__init__(self,infile.read(), param_ids='maxrep sortrec exprrec termrec'.split())

if __name__=='__main__':
  g=SMTLIBv2()
  sampler=GrammaSampler(g)
  sampler.update_params(maxrep=3, sortrec=.01, exprrec=.01, termrec=.001)
  while True:
    s=sampler.sample()
    if len(s)>100:
      break
  sys.stdout.buffer.write(s.encode('utf8','ignore'))
  print()


#!/usr/bin/env python3

from gramma import *

from collections import Counter
import os


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
  '''
    A gramma sideeffect that dumps a count of GRules occuring in each stack.

    It's used to identify rules that recurse too often.  E.g.

      when sampling from the grammar with a StackWatcher sideeffect attached:
        a:=['0'..'9'];
        r:=a|r;

      we would see the count of "r" skyrocket.  To control the size of sampled
      strings, we might limit its selection:
        a:=['0'..'9'];
        r:=a|.01 r;

  '''
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
    
    with open(os.path.join(os.path.dirname(__file__), 'SMTLIBv2.glf')) as infile:
      #GrammaGrammar.__init__(self,infile.read(), param_ids='maxrep sortrec exprrec termrec'.split(), sideeffects=[StackWatcher()])
      GrammaGrammar.__init__(self,infile.read(), param_ids='maxrep sortrec exprrec termrec'.split())

if __name__=='__main__':
  g=SMTLIBv2()
  sampler=GrammaSampler(g)
  sampler.update_params(maxrep=3, sortrec=.01, exprrec=.01, termrec=.001)
  s=sampler.sample()
  sys.stdout.buffer.write(s.encode('utf8','ignore'))
  print()


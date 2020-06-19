#!/usr/bin/env python3

from gramma import *

from collections import Counter
import os
from datetime import datetime,timedelta


class EveryNSeconds:
  def __init__(self, seconds):
    self.seconds=seconds
    self.reset()
  def reset(self):
    self.ready=datetime.now()+timedelta(seconds=self.seconds)
  def __bool__(self):
    if datetime.now()>=self.ready:
      self.reset()
      return  True
    return False
  __nonzero__=__bool__

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
  def __init__(self):
    self.every=EveryNSeconds(5)

  def reset_state(self,state):
    state.stk=[]

  def push(self,x,ge):
    if isinstance(ge,GRule):
      x.state.stk.append(ge.rname)
      c=Counter(x.state.stk)
      if self.every:
        print('============')
        for ruleName, count in c.most_common(5):
          print('  %8d %s' % (count, ruleName))
        print('------------')
      return True
    return False

  def pop(self,x,w,s):
    if w:
      del x.state.stk[-1]

class MyGrammar(GrammaGrammar):
  def __init__(self, grammarfile):
    with open(grammarfile) as infile:
      GrammaGrammar.__init__(self,infile.read(), param_ids='maxrep'.split())

  def reset_state(self,state):
    pass

  @gfunc
  def neg(x, n):
    yield 'neg(%s)' % str(n)

  @gfunc
  def eof(x):
    yield ''

  @gfunc
  def any(x):
    yield 'x'

  @gfunc
  def action(x,bodyg):
    body=bodyg.as_str()
    yield 'Action(%s)' % body

if __name__=='__main__':
  g=MyGrammar(sys.argv[1])
  sampler=GrammaSampler(g)
  sampler.update_params(maxrep=3)
  #sampler.add_sideeffects(StackWatcher())
  try:
    while True:
      s=sampler.sample()
      sys.stdout.buffer.write(s.encode('utf8','ignore'))
      sys.stdout.flush()
  except KeyboardInterrupt:
    pass

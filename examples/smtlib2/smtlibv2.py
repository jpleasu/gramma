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









class Defaulted(object):
    def __init__(self, default_value):
        self._default_=default_value

    def __getattr__(self, name):
        return self._default_

class Scoping(SideEffect):
    def reset_state(self,state):
        state.d=0
        state.scope_stack=[]
        state.scope=None

    def push(self,x,ge):
        if isinstance(ge, GRule) and ge.rname=='block':
            x.state.d+=1

            stk=x.state.scope_stack
            # initalize new scope with vars and has_vars
            sc=Defaulted(False)
            sc.vars=[]
            # if any parent has vars, so do we
            if len(stk)>0 and stk[-1].has_vars:
                sc.has_vars=True

            # update state
            x.state.scope=sc
            stk.append(sc)
            return True
        return False

    def pop(self,x,w,s):
        if w:
            x.state.scope_stack.pop()
            if len(x.state.scope_stack)>0:
                x.state.scope=x.state.scope_stack[-1]
            x.state.d-=1


def unwrapsym(sym):
  if sym.startswith('|'):
    return sym[1:-1]
  return sym

class Context:
  def __init__(self):
    self.fun2sort={}
    self.sort2funs={}

    self.sorts=[]

  def declareFun(self, sym, argsorts, retsort):
    #sym=unwrapsym(sym)
    sort=(argsorts, retsort)
    self.fun2sort[sym]=sort
    self.sort2funs[sort]=sym

class ContextStack:
  def __init__(self):
    self.stack=[Context()]

  def top(self):
    if len(self.stack)>0:
      return self.stack[-1]
    return Context()

  def push(self):
    self.stack.append(Context())

  def pop(self):
    if len(self.stack)>0:
      self.stack.pop()
    

class SMTLIBv2(GrammaGrammar):
  def __init__(self):
    with open(os.path.join(os.path.dirname(__file__), 'SMTLIBv2.glf')) as infile:
      #GrammaGrammar.__init__(self,infile.read(), param_ids='maxrep sortrec exprrec termrec'.split(), sideeffects=[StackWatcher()])
      GrammaGrammar.__init__(self,infile.read(), param_ids='maxrep sortrec exprrec termrec'.split())

  def reset_state(self,state):
    state.stack=ContextStack()
    state.sampler=GrammaSampler(self)

  @gfunc
  def push(x, n):
    x.state.stack.push()
    #yield (yield n)
    yield 'push 1'
  
  @gfunc
  def pop(x, n):
    x.state.stack.pop()
    #yield (yield n)
    yield 'pop 1'

  @gfunc
  def declare_fun(x, symg,  argsortsg, retsortg):
    sym=yield symg
    argsorts=yield argsortsg
    retsort=yield retsortg
    x.state.stack.top().declareFun(sym,argsorts,retsort)
    yield 'declare-fun %s (%s) %s' % (sym, argsorts, retsort)

if __name__=='__main__':
  g=SMTLIBv2()
  sampler=GrammaSampler(g)
  sampler.update_params(maxrep=3, sortrec=.01, exprrec=.01, termrec=.001)
  s=sampler.sample()
  sys.stdout.buffer.write(s.encode('utf8','ignore'))
  print()


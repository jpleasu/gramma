#!/usr/bin/env python
'''
use coroutines to sample

TODO
====
- should a tracetree record the choices made?
  - repeating an Alt node is choosing the same child
  - repeating a Rep node is choosing the same _number_ of repetitions
  - "nonempty(child)" gfunc?
    - no entropy, just sampling. If its children are fixed, it can return a
      constant (its original result), o/w is must be run again.
  - "randint()" gfunc?
    - repeating its behavior would be just returning the same string.. no gexpr
      args
  - "newvar(id)" and "randvar()" gfuncs?
    - newvar populates state, so it must be run
    - randvar randomly selects an existing state, so state must be prescribed
      and it must get the same choice.

- do constrained sampling
  - tree constraint from a tracetree
  - global parameter setting, where nodes of the expression tree get parameters
  - a combination, where nodes of the trace tree get parameters
- composition?
  - produce a tracetree from a constrained sample result

  - mkgen returns a coroutine that will ask for samples by yielding GExprs. We
    can compose mkgens with functional composition, interpreting GExprs as we go.

      def mkgen1(x,ge):
        if isinstance(ge,Alt):
          # do something interesting
        return default_mkgen(x,ge)

    the sampler is always calling the same mkgen.. although we could have
    progress through a tracetree..

  - compile to coroutines..
  - analyze coroutines to avoid 'yield' keyword within list comphrehensions
'''
from __future__ import absolute_import, division, print_function
import sys
if sys.version_info < (3,0):
    #from builtins import (bytes, str, open, super, range,zip, round, input, int, pow, object)
    
    # builtins' object fucks up slots
    from builtins import (bytes, open, super, range,zip, round, input, int, pow)
    import __builtin__
else:
    import builtins as __builtin__
    xrange=range



import sys
import types
import numpy as np
from collections import namedtuple

class GExpr(object):
  pass

class Tok(GExpr):
  def __init__(self,s):
    self.s=s

class Alt(GExpr):
  def __init__(self, *l):
    self.children=l

class Cat(GExpr):
  def __init__(self, *l):
    self.children=l


class Rep(GExpr):
  def __init__(self, sub, mi, ma):
    self.sub=sub
    self.mi=mi
    self.ma=ma

class Rule(GExpr):
  def __init__(self, rhs=None):
    self.rhs=rhs
  def set(self,rhs):
    self.rhs=rhs

class TTNode(object):
  __slots__='ge','parent','children','s'
  def __init__(self,ge):
    self.ge=ge
    self.parent=None
    self.children=[]
    self.s=None

  def set_parent(self,p):
    self.parent=p
    p.children.append(self)

  def dump(self,indent=0,out=sys.stdout):
    print('%s%s "%s"' % ('  '*indent, self.ge, self.s),file=out)
    for c in self.children:
      c.dump(indent+1,out)

class Sampler(object):
  def __init__(self):
    X=type('X', (), dict(__slots__=('random','state')))
    self.x=X()
    self.x.random=np.random.RandomState()
    self.x.state=type('',(),{})

  def reset(self):
    self.x.state=type('',(),{})

  def itorstr(self,ge):
    if isinstance(ge,GExpr):
      # is a child of one up (or root)
      # enter
      return self.mkgen(ge)
    else:
      # fed to two up
      # exit
      return ge

  def sample(self,ge):
    self.stack=[self.itorstr(ge)]
    while True:
      if isinstance(self.stack[-1],str):
        s=self.stack.pop()
        if len(self.stack)==1:
          return s
        self.stack[-1]=self.itorstr(self.stack[-2].send(s))
      else:
        self.stack.append(self.itorstr(next(self.stack[-1])))

  def mkgen(self,ge):
    if isinstance(ge,Tok):
      def g():
        yield ge.s
    elif isinstance(ge,Alt):
      def g():
        s=yield self.x.random.choice(ge.children)
        yield s
    elif isinstance(ge,Cat):
      def g():
        s=''
        for cge in ge.children:
          s+=yield cge
        yield s
    elif isinstance(ge,Rule):
      def g():
        s=yield(ge.rhs)
        yield s
    elif isinstance(ge,Rep):
      def g():
        s=''
        n=self.x.random.randint(ge.mi,ge.ma+1)
        while n>0:
          s+=yield ge.sub
          n-=1
        yield s
    else:
      raise Exception('unrecognized GExpr type: %s' % type(ge))
    return g()

class TraceTreeVisitorSampler(Sampler):
  def reset(self):
    super().reset()
    self.tt_root=None
    self.tt=None

  def itorstr(self,ge):
    if isinstance(ge,GExpr):
      # is a child of one up (or root)
      n=TTNode(ge)
      if self.tt==None:
        self.tt_root=n
      else:
        n.set_parent(self.tt)
      self.tt=n

      return self.mkgen(ge)
    else:
      self.tt.s=ge
      self.tt=self.tt.parent
      # fed to two up
      return ge


if __name__=='__main__':
  r=Rule()
  r.set(Alt(Rep(Tok(">"),1,3), Cat(Tok("."),r)))

  ge=r

  s=TraceTreeVisitorSampler()
  for i in range(10):
    s.reset()
    print('=============')
    print(s.sample(ge))
    print('-------------')
    s.tt_root.dump()


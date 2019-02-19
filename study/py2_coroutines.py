#!/usr/bin/env python
'''
use coroutines to sample

TODO
====
- can a tracetree record the choices made?
- do constrained sampling
  - tree constraint from a tracetree
  - global parameter setting, where nodes of the expression tree get parameters
  - a combination, where nodes of the trace tree get parameters
- composition?
  - produce a tracetree from a constrained sample result

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


class Rule(GExpr):
  def __init__(self, rhs=None):
    self.rhs=rhs
  def set(self,rhs):
    self.rhs=rhs

def mkgen(ge):
  if isinstance(ge,Tok):
    def g(x):
      yield ge.s
  elif isinstance(ge,Alt):
    def g(x):
      s=yield x.random.choice(ge.children)
      yield s
  elif isinstance(ge,Cat):
    def g(x):
      s=''
      for cge in ge.children:
        s+=yield cge
      yield s
  elif isinstance(ge,Rule):
    def g(x):
      s=yield(ge.rhs)
      yield s
  else:
    raise Exception('unrecognized GExpr type: %s' % type(ge))
  return g


def sample(x,ge):
  'recursive sampler'

  g=mkgen(ge)(x)
  r=next(g)
  while not isinstance(r,str):
    r=g.send(sample(x,r))
  return r

def itorstr(x,ge):
  print('||| %s |||' % ge)
  if isinstance(ge,GExpr):
    return mkgen(ge)(x)
  else:
    return ge

def stk_sample(x,ge):
  '''
    a sampler that doesn't rely on recursion
  '''
  stack=[itorstr(x,ge)]
  while True:
    if isinstance(stack[-1],str):
      # stack looks like
      #   caller     <- parent generator, awaiting result
      #     callee   <- child generator, now finished
      #       result <- its string result
      #
      #  so send the result into caller, and push caller's next
      s=stack.pop()
      if len(stack)==1:
        return s
      stack[-1]=itorstr(x,stack[-2].send(s))
    else:
      it=stack[-1]
      stack.append(itorstr(x,next(it)))


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
      return self.mkgen(ge)
    else:
      # fed to two up
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
    else:
      raise Exception('unrecognized GExpr type: %s' % type(ge))
    return g()

class TraceTreeVisitorSampler(Sampler):
  def reset(self):
    super().reset()
    self.tt=[]
    self.tt_root=None

  def itorstr(self,ge):
    if isinstance(ge,GExpr):
      # is a child of one up (or root)
      n=TTNode(ge)
      if len(self.tt)==0:
        self.tt_root=n
      else:
        n.set_parent(self.tt[-1])
      self.tt.append(n)

      return self.mkgen(ge)
    else:
      n=self.tt.pop()
      n.s=ge
      # fed to two up
      return ge


def basic_test():
  r=Rule()
  r.set(Alt(Tok(">"), Cat(Tok("."),r)))
  
  #ge=Alt(Tok('asdf'),Tok('1234'))
  
  ge=r
  
  X=namedtuple('X','random state')
  x=X(np.random.RandomState(),type('',(),{}))
  
  for i in range(10):
    #print(sample(x,ge))
    print(stk_sample(x,ge))

if __name__=='__main__':
  r=Rule()
  r.set(Alt(Tok(">"), Cat(Tok("."),r)))

  ge=r

  s=TraceTreeVisitorSampler()
  for i in range(10):
    s.reset()
    print('=============')
    print(s.sample(ge))
    print('-------------')
    s.tt_root.dump()


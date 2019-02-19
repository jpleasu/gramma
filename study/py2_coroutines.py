#!/usr/bin/env python
'''
use coroutines to sample

TODO
====
- produce a tracetree
- do constrained sampling
  - tree constraint from a tracetree
  - global parameter setting, where nodes of the expression tree get parameters
  - a combination, where nodes of the trace tree get parameters
- composition?
  - produce a tracetree from a constrained sample result

'''
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

r=Rule()
r.set(Alt(Tok(">"), Cat(Tok("."),r)))

#ge=Alt(Tok('asdf'),Tok('1234'))

ge=r

X=namedtuple('X','random state')
x=X(np.random.RandomState(),type('',(),{}))

for i in range(10):
  #print(sample(x,ge))
  print(stk_sample(x,ge))


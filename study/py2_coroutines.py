#!/usr/bin/env python
'''
use coroutines to sample

TODO
====
- everything would return a function that builds generators, o/w they're not
  reusable

'''
import types
import random

def t(s):
  yield s

def c():
  yield (yield t('c'))


def alt(*l):
  g=random.choice(l)
  yield (yield g)

def cat(*l):
  # this works in python2, not 3..
  #yield ''.join([(yield(g)) for g in l])
  ll=[]
  for g in l:
    ll.append((yield(g)))
  yield ''.join(ll)

def randnum():
  yield ''.join(random.choice('0123456789') for i in range(random.randrange(1,5)))

def recurs():
  g=alt(t('>'),cat(t('-'), recurs()))
  yield (yield g)


def toit(fg):
  if isinstance(fg,str):
    return fg
  elif isinstance(fg,types.FunctionType):
    return fg()
  else:
    return iter(fg)


def sample(ge):
  'recursive sampler'
  ge=toit(ge)
  r=next(ge)
  while not isinstance(r,str):
    r=ge.send(sample(r))
  return r

class StackNode:
  __slots__='g','v'
  def __init__(self, g):
    self.g=toit(g)
    self.v=None


def stk_sample(g):
  'stacked sampler'
  stack=[toit(g)]
  while True:
    #print('%3d %s' % (len(stack), stack))
    n=stack[-1]
    if isinstance(n,str):
      # stack looks like
      #   caller     <- parent generator, awaiting result
      #     callee   <- child generator, now finished
      #       result <- its string result
      stack.pop()
      if len(stack)==1:
        return n
      stack[-1]=toit(stack[-2].send(n))
    else:
      stack.append(next(n))



def mkref(d,name):
  def ref(d=d):
    yield (yield d[name]()) 
  return ref

for i in range(10):
  #g=cat(randnum(), alt(t('opt1'),t('opt2')), alt(t('opt3'),t('opt4')))
  #g=cat(t('<'), recurs)
  #g=c
  #g=cat(t('a'),t('b'))
  
  d={}
  d['r']=lambda: alt(t('>'), cat(t('-'),mkref(d,'r')))
  g=d['r']()


  print(sample(g))
  #print(stk_sample(g))
  #print(stk_sample(g))


#!/usr/bin/env python3

import sys,os
import z3
from smtlibv2 import *

import re
import traceback

def dump_node(n):
  d=0
  while n!=None:
      print('   %s%s' % (' '*d, n.ge))
      n=n.parent

from collections import namedtuple
Err=namedtuple('Err', 'msg line column'.split())
def parse_z3_error(msg):
  m=re.search(r'error "line (\d+) column (\d+)', msg)
  if m!=None:
    return Err(msg[m.end():], int(m.group(1)), int(m.group(2)))
  return Err(msg, None, None)

def parse_z3_error0(v):
  v.msg=v.value.decode('utf8')
  m=re.search(r'error "line (\d+) column (\d+)', v.msg)
  if m!=None:
    v.line,v.column=int(m.group(1)), int(m.group(2))
    return True
  v.line,v.column=None,None
  return False


def readitall(fd):
  BUFFER_SIZE=1024
  buf=b''
  while True:
    try:
      b=os.read(fd, BUFFER_SIZE)
      if len(b)==0:
        break
      buf += b
    except OSError as err:
      if err.errno == errno.EAGAIN or err.errno == errno.EWOULDBLOCK:
        break
      else:
        raise err
  return buf
 
def withtrace():
  g=SMTLIBv2()
  sampler=GrammaSampler(g)
  sampler.update_params(maxrep=3, sortrec=.9, exprrec=.5, termrec=.5)
  tracer=Tracer()
  sampler.add_sideeffects(tracer)
  fifopath='fofum.%d' % os.getpid()
  os.mkfifo(fifopath)
  cmd_setup='(set-option :diagnostic-output-channel "'+fifopath+'")'
  try:
    fofum=os.open(fifopath,os.O_RDONLY|os.O_NONBLOCK)
    i=0
    while True:
      i+=1
      print('= %d =' % i)
    
      while True:
        samp=sampler.sample()
        if len(samp)>100:
          break
      print(samp)
      sol=z3.Solver()
      try:
        sol.from_string(cmd_setup+samp)
        print('  result = %s' % sol.check())
      except KeyboardInterrupt:
        break
      except z3.z3types.Z3Exception as v:
        err=readitall(fofum).decode('utf8').strip()
        if len(err)>0:
          print('err: %s' % err.strip())

        for line in v.value.decode('utf8').strip().split('\n'):
          e=parse_z3_error(line)
          if e.column!=None:
            col=e.column-1-len(cmd_setup)
            print(' '*col+'^ - '+e.msg.strip())
            n=tracer.tracetree.child_containing(col)
            dump_node(n)
          else:
            print(e.msg.strip())
      except:
        traceback.print_exc()
        break
  finally:
    os.unlink(fifopath)

from datetime import datetime, timedelta
class Ticker:
  def reset(self):
    self.done=datetime.now()+timedelta(seconds=self.seconds)
  def __init__(self, seconds):
    self.seconds=seconds
    self.reset()
  def __bool__(self):
    t = datetime.now()>self.done
    if t:
      self.reset()
    return t
  __nonzero__=__bool__

def spam():
  g=SMTLIBv2()
  sampler=GrammaSampler(g)
  sampler.update_params(maxrep=3, sortrec=.99, exprrec=.99, termrec=.99)
  pid=os.getpid()
  while True:
    while True:
      samp=sampler.sample()
      if len(samp)>100:
        break
    with open('samp.%d.txt' % pid, 'w') as out:
      out.write(samp)
    sol=z3.Solver()
    try:
      sol.from_string(samp)
      sol.check()
    except KeyboardInterrupt:
      break
    except z3.z3types.Z3Exception as v:
      pass
    except:
      traceback.print_exc()
      break

spam()
#withtrace()

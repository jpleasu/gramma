#!/usr/bin/env python3

import sys
import z3
from smtlibv2 import *

import re
import traceback

def dump_node(n):
  d=0
  while n!=None:
      print('   %s%s' % (' '*d, n.ge))
      n=n.parent
      d+=1

def parse_z3_error(v):
  v.msg=v.value.decode('utf8')
  m=re.search(r'error "line (\d+) column (\d+)', v.msg)
  if m!=None:
    v.line,v.column=int(m.group(1)), int(m.group(2))
  else:
    v.line,v.column=None,None

g=SMTLIBv2()
sampler=GrammaSampler(g)
sampler.update_params(maxrep=3, sortrec=.01, exprrec=.1, termrec=.01)
#sys.stdout.buffer.write(samp.encode('utf8','ignore'))
tracer=Tracer()
sampler.add_sideeffects(tracer)

i=0
while True:
  i+=1
  print('= %d =' % i)

  samp=sampler.sample()
  print(samp)
  print(' -')
  sol=z3.Solver()
  try:
    sol.from_string(samp)
    print(sol.check())
  except KeyboardInterrupt:
    break
  except z3.z3types.Z3Exception as v:
    parse_z3_error(v)
    n=tracer.tracetree.child_containing(v.column-1)
    print(v.msg)
    dump_node(n)
  except:
    traceback.print_exc()
print(sol)



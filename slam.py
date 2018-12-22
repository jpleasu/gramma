#!/usr/bin/env python
import StringIO
import subprocess
import sys
from itertools import islice

import gen_greatview as greatview

errcnt = 0
segfault = 0
def slam_one(binary, input):
    global errcnt,segfault
    input = input + '\x00'
    #input=open('../greatview/pov/pov','rb').read()
    p = subprocess.Popen(binary, shell=True, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    o,e = p.communicate(input)
    if 'Segmentation fault' in e:
        segfault += 1
    elif p.returncode != 0:
        errcnt += 1
        #print '----'
        #print input


tot = 0
g = greatview.Greatview(sys.argv[2])
for st,x in islice(g.generate(), 1000):
    if len(x) > 0:
        tot+=1
        slam_one(sys.argv[1], x)

if segfault>0:
    print '*******************************************'
print 'tot=%d' % tot
print 'errcnt=%d' % errcnt
print 'segfault=%d' % segfault

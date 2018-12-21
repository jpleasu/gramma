#!/usr/bin/env python
import StringIO
import subprocess
import sys
from itertools import islice

import gen_greatview as greatview

cnt = 0
def slam_one(binary, input):
    global cnt
    input = input + '\x00'
    fake_fd = StringIO.StringIO(input)
    p = subprocess.Popen(binary, shell=True, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    r = p.communicate(fake_fd.read())
#    print p.returncode
    if p.returncode != 0:
        cnt += 1


g = greatview.Greatview(sys.argv[2])
for x in islice(g.generate(), 100):
    if len(x) > 0:
        slam_one(sys.argv[1], x)

print cnt

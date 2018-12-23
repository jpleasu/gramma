#!/usr/bin/env python
from os import environ
import StringIO
import subprocess
import sys
from itertools import islice
from sysv_ipc import SharedMemory, IPC_PRIVATE, IPC_CREAT, IPC_EXCL

import gen_greatview as greatview

cnt = 0
class Slammer():
    def __init__(self, binary):
        self.binary = binary
        self.winners = 0

        self._bin_table = bytearray(256)
        self._bin_table[0] = 0
        self._bin_table[1] = 1
        for x in range(2,4):
            self._bin_table[x] = 2
        for x in range(4,8):
            self._bin_table[x] = 4
        for x in range(8,16):
            self._bin_table[x] = 8
        for x in range(16,32):
            self._bin_table[x] = 16
        for x in range(32,64):
            self._bin_table[x] = 32
        for x in range(64,128):
            self._bin_table[x] = 64
        for x in range(128,256):
            self._bin_table[x] = 128
        self._trace_bits = bytearray(65536)
        self._shm = SharedMemory(IPC_PRIVATE, IPC_CREAT | IPC_EXCL, 0600, 65536)
        self._shm.detach()
        environ['__AFL_SHM_ID'] = str(self._shm.id)
>>>>>>> slam now counts uniq afl bitmaps

        self._hits = {}

<<<<<<< 8ff333a80a3dafd9a40792aec5cc44bc44db1651
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
=======
    def slam_one(self, input):
        input = input + '\x00'
        fake_fd = StringIO.StringIO(input)
        p = subprocess.Popen(self.binary, shell=True, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        r = p.communicate(fake_fd.read())
        if p.returncode == 0:
            self.winners += 1
            
    def afl_one(self, input):
        self._shm.attach()
        self._shm.write('\x00'*65536)
        self.slam_one(input)

        # todo: this is SLOWWWWWW. Use carter's approach instead
        for i,b in enumerate(self._shm.read()):
            b = ord(b)
            self._trace_bits[i] = self._bin_table[b]

        self._shm.detach()
        h = hash(str(self._trace_bits))
        if h in self._hits:
            self._hits[h] += 1
        else:
            self._hits[h] = 1

slammer = Slammer(sys.argv[1])
g = greatview.Greatview(sys.argv[2])

#slammer.afl_one('new num a 20')
#exit(0)

tests = 0
for x in g.generate():
    if len(x) > 0:
        slammer.afl_one(x)
        # slammer.slam_one(x)
        if tests >= 1000:
            break
        tests += 1

print 'cnt:',tests, 'uniq:', len(slammer._hits), 'winners:', slammer.winners

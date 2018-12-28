#!/usr/bin/env python
import numpy as np
from os import environ
import StringIO
import subprocess
import sys
from itertools import islice
from sysv_ipc import SharedMemory, IPC_PRIVATE, IPC_CREAT, IPC_EXCL

from gramma import *
import gen_greatview as greatview

cnt = 0
class Slammer():
    def __init__(self, binary):
        self.binary = binary
        self.errcnt = 0
        self.segfaults = 0

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

        self._shm = SharedMemory(IPC_PRIVATE, IPC_CREAT | IPC_EXCL, 0600, 65536)
        environ['__AFL_SHM_ID'] = str(self._shm.id)

        self._virgin_bits = bytearray('\xff' * 65536)
        self._cov = 0
        self._cnt = 0

        self.tests = 0
        
    def _stats(self):
        if self.tests % 100 == 0 and self.tests <> 0:
            print 'cnt:',self.tests, 'cov:', self._cov, 'cnt:', self._cnt, 'errcnt:', self.errcnt, 'segfaults:', self.segfaults
            sys.stdout.flush()
        
    def slam_one(self, input):
        if len(input) == 0:
            return
        self._stats()
        self.tests += 1
            
        input = input + '\x00'
        fake_fd = StringIO.StringIO(input)
        p = subprocess.Popen(self.binary, shell=True, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        (o,e) = p.communicate(fake_fd.read())
        if 'Segmentation fault' in e:
            self.segfaults += 1
        elif p.returncode != 0:
            self.errcnt += 1
            
    def afl_one(self, input):
        if len(input) == 0:
            return
        self._shm.write('\x00'*65536)
        self.slam_one(input)

        # todo: this is SLOWWWWWW. Use carter's approach instead
        x = self._shm.read()
        cov = 0
        for i,b in enumerate(self._shm.read()):
            b = ord(b)
            b = self._bin_table[b]
            vb = self._virgin_bits[i]
            v = (vb & ~b) & 0xff
            if v <> vb:
                if vb == 0xff:
                    cov = 1
                elif cov == 0:
                    cov = 2
            self._virgin_bits[i] = v

        if cov == 1:
            self._cov += 1
        elif cov == 2:
            self._cnt += 1

        return cov

slammer = Slammer(sys.argv[1])
g = greatview.Greatview(sys.argv[2])

#slammer.afl_one('new num a 20')
#exit(0)

tests = 0

def no_resample():
    global tests
    for (st, x) in g.generate():
        #slammer.slam_one(x)
        slammer.afl_one(x)

def use_resample():
    while True:
        x = g.build_richsample(np.random.get_state())
        progress = slammer.afl_one(x.s)
        progress = 0
        if progress:
            nesting_nodes=[rr for rr in x.genwalk() if isinstance(rr.gt,GRule) and rr.gt.rname=='nesting']
            node = np.random.choice(nesting_nodes)
            saved_state = node.inrand
            node.inrand = None
            for s in islice(g.gen_resamples(x),20):
                slammer.afl_one(s)
            node.inrand = saved_state
            
#        if slammer.tests > 1000000:
#            break
        
use_resample()

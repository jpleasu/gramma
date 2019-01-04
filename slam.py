#!/usr/bin/env python
from ctypes import CDLL, pointer, POINTER, Structure, c_uint, c_ubyte

import numpy as np
from os import environ
import StringIO
import subprocess
import sys
from itertools import islice
from sysv_ipc import SharedMemory, IPC_PRIVATE, IPC_CREAT, IPC_EXCL

from gramma import *
import gen_greatview as greatview

def print_syndrome(prefix, syndrome):
    print prefix,
    for (i,b) in syndrome:
        print '(%d %x)' % (i,b),
    print

class syndrome_el(Structure):
    _fields_ = [('idx', c_uint),
               ('v', c_ubyte)]
    
class syndrome(Structure):
    _fields_=[('len', c_uint),
             ('els', syndrome_el*65536)] 

class queue_el():
    def __init__(self, cksum, syndrome):
        self.cksum = cksum
        self.syndrome = syndrome
        self.nfuzz = 1
        
class Slammer():
    def __init__(self, binary, native=True):
        self.binary = binary
        self.errcnt = 0
        self.segfaults = 0
        self._native = native
        self._hashes = {}
        self._queue = []

        if native:
            self._afl = CDLL('afl.so')
            environ['__AFL_SHM_ID'] = str(self._afl.init())
            self._afl.get_syndrome.restype = POINTER(syndrome)
            self._afl.get_trace_bits.restype = POINTER(c_ubyte)
        else:
            self._bin_table = bytearray(256)
            self._bin_table[0] = 0
            self._bin_table[1] = 1
            self._bin_table[2] = 2
            self._bin_table[3] = 4
            for x in range(4,8):
                self._bin_table[x] = 8
            for x in range(8,16):
                self._bin_table[x] = 16
            for x in range(16,32):
                self._bin_table[x] = 32
            for x in range(32,127):
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
        singles = 0
        doubles = 0
        for el in self._queue:
            if el.nfuzz == 1:
                singles += 1
            elif el.nfuzz == 2:
                doubles += 1

        l = len(self._queue)
        exp_total_paths = l
        if doubles > 0:
            exp_total_paths += (singles * singles) / (2 * doubles)
        else:
            exp_total_paths += (singles * (singles - 1)) / 2

        if l > 0:
            correctness = float(singles) / self.tests
        else:
            correctness = 0

        if self.tests % 100 == 0 and self.tests <> 0:
            print 'cnt:',self.tests, 'cov:', self._cov, 'cnt:', self._cnt, 'errcnt:', self.errcnt, 'segfaults:', self.segfaults, 'exp paths:', exp_total_paths, 'correctness:', correctness
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
        if self._native:
            self._afl.clear_trace()
        else:
            self._shm.write('\x00'*65536)
        
        self.slam_one(input)

        syndrome = []
        if self._native:
            res = self._afl.has_new_bits()
            s = self._afl.get_syndrome().contents
            for i in xrange(0,s.len):
                e = s.els[i]
                syndrome.append((e.idx, e.v))
            trace_bits = self._afl.get_trace_bits()
        else:
            res = 0
            trace_bits = bytearray(self._shm.read())
            for i,b in enumerate(trace_bits):
                trace_bits[i] = self._bin_table[b]
                
            for i,b in enumerate(trace_bits):
                vb = self._virgin_bits[i]
                v = (vb & ~b) & 0xff
                if v <> vb:
                    syndrome.append((i, vb & b))
                    if vb == 0xff:
                        res = 2
                    elif res == 0:
                        res = 1
                self._virgin_bits[i] = v

        if res == 2:
            self._cov += 1
        elif res == 1:
            self._cnt += 1


        '''
        if self._native:
            cksum = self._afl.hash()
        else:
            cksum = hash(str(self._shm.read()))

        for el in self._queue:
            if el.cksum == cksum:
                el.nfuzz += 1

        for el in self._queue:
            if el.cksum == cksum:
                el.nfuzz += 1
        '''

        for el in self._queue:
            same = True
            for x in el.syndrome:
                (i, b) = x
                if trace_bits[i] & b == 0:
                    same = False
                    break

            if same:
                el.nfuzz += 1
            
        cksum = 0
        if res:
            self._queue.append(queue_el(cksum, syndrome))

        return res

#np.random.seed(1)

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
        # progress = slammer.afl_one('new num a 10\n\x00')
        # exit(0)
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

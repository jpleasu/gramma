#!/usr/bin/env python
import pdb
import struct

from gramma import *

class Cml(Gramma):
    def __init__(x, path):
        with open(path) as infile:
            Gramma.__init__(x,infile.read())
   
    @gfunc
    def size(x,child):
        n=x.sample(child)
        return struct.pack('>H', len(n)) + n

    @gfunc
    def el_builder(x, left, id, attrs, data, right):
        l = x.sample(left)
        i = x.sample(id)
        a = x.sample(attrs)
        d = x.sample(data)
        r = x.sample(right)
        return l + i + a + r + d + l + '/' + i + r
    
if __name__=='__main__':
    g=Cml('cml.glf')

    #np.random.seed(1)
    for st,x in g.generate():
        if len(x)>0:
            print('----')
            print(x)

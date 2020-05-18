#!/usr/bin/env python
import pdb
import struct

from gramma import *

class Cml(GrammaGrammar):
    def __init__(x, path):
        with open(path) as infile:
            GrammaGrammar.__init__(x,infile.read(), sideeffects=[GeneralDepthTracker(lambda ge:isinstance(ge,GRule) and ge.rname=='element')])

    @gfunc
    def size(x,child):
        n=yield child
        yield struct.pack('>H', len(n)) + n

    @gfunc
    def el_builder(x, left, id, attrs, data, right):
        l = yield left
        i = yield id
        a = yield attrs
        d = yield data
        r = yield right
        yield l + i + a + r + d + l + '/' + i + r
    
if __name__=='__main__':
    g=Cml('cml.glf')
    sampler=GrammaSampler(g)

    while True:
        x=sampler.sample()
        if len(x)>0:
            print('----')
            print(x)

# vim: ts=4 sw=4

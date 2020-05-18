#!/usr/bin/env python
from gramma import *

import pdb

import random

class Greatview(GrammaGrammar):
    def __init__(x, gvglf_path):
        with open(gvglf_path) as infile:
            GrammaGrammar.__init__(x,infile.read(), sideeffects=[GeneralDepthTracker(lambda ge:isinstance(ge,GRule) and ge.rname=='nesting')])
   
    def reset_state(self,state):
        super().reset_state(state)

        state.idstack=[]

        state.lastval=None
        state.exp=None
        state.remembered={}

    @gfunc
    def rand_val(x):
        x.state.exp=x.random.choice([8,16,32])
        x.state.lastval=x.random.randint(0,2**x.state.exp-1)
        yield '%d' % x.state.lastval

    @gfunc
    def bigger_val(x):
        v=x.random.randint(x.state.lastval+1,2**x.state.exp)
        yield '%d' % v

    @gfunc
    def old(x,namespace):
        namespace=namespace.as_str()
        yield x.random.choice(list(x.state.remembered.get(namespace)))
        
    @gfunc
    def new(x,namespace,child):
        namespace=namespace.as_str()
        names=x.state.remembered.setdefault(namespace,set())
        n=yield child
        while n in names:
            n=yield child
        names.add(n)
        yield n

    @gfunc
    def ifdef(x, namespace, child):
        namespace=namespace.as_str()
        if len(x.state.remembered.get(namespace,[]))!=0:
            yield (yield child)
        yield ''

    @gfunc
    def nonempty(x,child):
        while True:
            s=yield child
            if len(s)!=0:
                yield s
    @gfunc
    def push(x,child):
        n=yield child
        x.state.idstack.append(n)
        yield n

    @gfunc
    def peek(x):
        yield x.state.idstack[-1]

    @gfunc
    def pop(x):
        x.state.idstack.pop()
        yield ''


def gensamples():
    g=Greatview('gv2.glf')
    sampler=GrammaSampler(g)
    sampler.random.seed(1)
    while True:
        x=sampler.sample()
        if len(x)>0:
            print('----')
            print(x)

def resampletest():
    global n
    g=Greatview('gv2.glf')
    sampler=GrammaSampler(g)
    #sampler.random.seed(4)
    tracer=Tracer()
    sampler.add_sideeffects(tracer)

    while True:
        s=sampler.sample()
        tt=tracer.tracetree
        news=[n for n in tt.gennodes() if isinstance(n.ge,GFunc) and n.ge.fname=='new']
        olds=[n for n in tt.gennodes() if isinstance(n.ge,GFunc) and n.ge.fname=='old']
        if len(olds)>3 and len(news)>3:
            break
    print('== original sample ==')
    print(s)

    #print('~~ tracetree ~~')
    #tt.dump()

    # resample all variable name constructions
    def randomize_news(rr):
        if isinstance(rr.ogt, GFunc) and rr.ogt.fname=='new':
            rr.inrand=None
    #r.visit(randomize_news)

    # resample every variable selection
    def randomize_olds(rr):
        if isinstance(rr.ogt, GFunc) and rr.ogt.fname=='old':
            rr.inrand=None
    #r.visit(randomize_olds)

    # resample integer generation
    def randomize_ints(rr):
        if isinstance(rr.ogt, GRule) and rr.ogt.rname=='int':
            rr.inrand=None
    #r.visit(randomize_ints)

    #nesting_nodes=[rr for rr in r.genwalk() if isinstance(rr.ogt,GRule) and rr.ogt.rname=='nesting']
    #nesting_nodes=[rr for rr in r.genwalk() if isinstance(rr.ogt,GRule) and rr.ogt.rname=='oneline']


    # resample a variable name
    #n=random.choice([n for n in tt.gennodes() if isinstance(n.ge,GFunc) and n.ge.fname=='new'])

    # resample a variable reference
    # guaranteeing that there's more than one old id available takes work..
    options=[n for n in tt.gennodes() if \
            isinstance(n.ge,GFunc) \
            and n.ge.fname=='old' \
            and len(n.instate.remembered[n.ge.fargs[0].as_str()])>1]
    print(options)
    n=random.choice(options)

    print('-- resampling --')
    print('"%s" at depth(n) = %d' % (n.ge,n.depth()))
    
    ## construct a GExpr that resamples only n
    rge,cfg=tt.resample(g,lambda t:t==n)
    print('-- the resample expression --')
    print(rge)
    
    sampler.update_cache(cfg)

    #print('-- meta --')
    #rge.dump_meta()

    for i in range(10):
        print('---')
        print(sampler.sample(rge))

if __name__=='__main__':
    #gensamples()
    resampletest()

# vim: ts=4 sw=4

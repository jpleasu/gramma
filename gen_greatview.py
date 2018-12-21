#!/usr/bin/env python
import pdb

from gramma import *

class Greatview(Gramma):
    def __init__(x):
        with open('gv2.glf') as infile:
            Gramma.__init__(x,infile.read())
        x.remembered={}
   
    def reset(x):
        Gramma.reset(x)
        x.idstack=[]

        x.lastval=None
        x.exp=None
        x.remembered.clear()

    @gfunc
    def rand_val(x):
        x.exp=np.random.choice([8,16,32])
        x.lastval=np.random.randint(0,2**x.exp-1)
        return '%d' % x.lastval

    @gfunc
    def bigger_val(x):
        v=np.random.randint(x.lastval+1,2**x.exp)
        return '%d' % v

    @gfunc
    def old(x,namespace):
        namespace=namespace.as_str()
        return np.random.choice(list(x.remembered.get(namespace)))
        
    @gfunc
    def new(x,namespace,child):
        namespace=namespace.as_str()
        names=x.remembered.setdefault(namespace,set())
        n=x.sample(child)
        while n in names:
            n=x.sample(child)
        names.add(n)
        return n

    @gfunc
    def ifdef(x, namespace, child):
        namespace=namespace.as_str()
        if len(x.remembered.get(namespace,[]))!=0:
            return x.sample(child)
        return ''

    @gfunc
    def push(x,child):
        n=x.sample(child)
        x.idstack.append(n)
        return n

    @gfunc
    def peek(x):
        return x.idstack[-1]

    @gfunc
    def pop(x):
        x.idstack.pop()
        return ''



def gensamples():
    g=Greatview()
    
    np.random.seed(1)
    for st,x in g.generate():
        if len(x)>0:
            print('----')
            print(x)

def resampletest():
    global r,x
    x=Greatview()
    np.random.seed(1)
    r=x.buildresampler()
    print('====')
    print(r.s)

    # resample all variable name constructions
    def randomize_news(rr):
        if isinstance(rr.gt, GFunc) and rr.gt.fname=='new':
            rr.inrand=None
    r.visit(randomize_news)

    # resample every variable selection
    def randomize_olds(rr):
        if isinstance(rr.gt, GFunc) and rr.gt.fname=='old':
            rr.inrand=None
    r.visit(randomize_olds)

    # resample integer generation
    def randomize_ints(rr):
        if isinstance(rr.gt, GFunc) and rr.gt.fname=='int':
            for rrr in rr.genwalk():
                rrr.inrand=None
    r.visit(randomize_ints)

    # resample one of the 'nesting' rule nodes
    nesting_nodes=[rr for rr in r.genwalk() if isinstance(rr.gt,GRule) and rr.gt.rname=='nesting']
    np.random.seed(2)
    np.random.choice(nesting_nodes).inrand=None
 

    for s in islice(x.gen_resamples(r),20):
        print('----')
        print(s)

resampletest()
#gensamples()

# vim: ts=4 sw=4

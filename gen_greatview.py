#!/usr/bin/env python
import pdb

from gramma import *

class Greatview(Gramma):
    def __init__(x):
        with open('gv2.glf') as infile:
            Gramma.__init__(x,infile.read())
        x.remembered={}

    def rand_val(x):
        x.exp=np.random.choice([8,16,32])
        x.lastval=np.random.randint(0,2**x.exp-1)
        return '%d' % x.lastval

    def bigger_val(x):
        v=np.random.randint(x.lastval+1,2**x.exp)
        return '%d' % v

    def old(x,namespace):
        namespace=x.getstring(namespace)
        return np.random.choice(list(x.remembered.get(namespace)))
        
    def new(x,namespace,child):
        namespace=x.getstring(namespace)
        names=x.remembered.setdefault(namespace,set())
        n=x.sample(child)
        while n in names:
            n=x.sample(child)
        names.add(n)
        return n

    def ifdef(x, namespace, child):
        namespace=x.getstring(namespace)
        if len(x.remembered.get(namespace,[]))!=0:
            return x.sample(child)
        return ''

    def push(x,child):
        n=x.sample(child)
        x.idstack.append(n)
        return n

    def peek(x):
        return x.idstack[-1]

    def pop(x):
        x.idstack.pop()
        return ''

   
    def reset(x):
        Gramma.reset(x)
        x.idstack=[]

        x.lastval=None
        x.exp=None
        x.remembered.clear()

def gensamples():
    g=Greatview()
    
    np.random.seed(1)
    for x in g.generate():
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

    def randomize_news(rr):
        ''' choose different variables'''
        if rr.et.name==u'func' and etfunc(rr.et).name=='new':
            rr.inrand=None
    r.visit(randomize_news)

    def randomize_olds(rr):
        ''' choose different variables'''
        if rr.et.name==u'func' and etfunc(rr.et).name=='old':
            rr.inrand=None
    r.visit(randomize_olds)

    def randomize_ints(rr):
        ''' different random integers'''
        if rr.et.name==u'rule' and etrule(rr.et).name=='int':
            for rrr in rr.genwalk():
                rrr.inrand=None
    r.visit(randomize_ints)

    for s in islice(x.gen_resamples(r),20):
        print('----')
        print(s)

resampletest()
#gensamples()

# vim: ts=4 sw=4

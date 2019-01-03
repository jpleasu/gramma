#!/usr/bin/env python
import pdb

from gramma import *

class Greatview(Gramma):
    def __init__(x, gvglf_path):
        with open(gvglf_path) as infile:
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
    def nonempty(x,child):
        while True:
            s=x.sample(child)
            if len(s)!=0:
                return s
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
    g=Greatview('gv2.glf')
    
    np.random.seed(1)
    for st,x in g.generate():
        if len(x)>0:
            print('----')
            print(x)

def resampletest():
    global r,x
    x=Greatview('gv2.glf')
    x.random.seed(3)
    while True:
        r=x.build_richsample(np.random.get_state())
        if len(r.s)>0:
            ss=x.generate(r).next()[1]
            if r.s!=ss:
                print('====')
                print(r.s)
                print('~~~~')
                print(ss)

def resampletest2():
    global r,x
    x=Greatview('gv2.glf')
    x.random.seed(4)
    while True:
        rseed=np.random.get_state()
        r=x.build_richsample(rseed)
        if len(r.s)>0:
            break
    print('====')
    print(r.s)
    print('~~~~')
    print(r)


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


    # resample one of the 'nesting' rule nodes
    nesting_nodes=[rr for rr in r.genwalk() if isinstance(rr.ogt,GRule) and rr.ogt.rname=='nesting']
    #nesting_nodes=[rr for rr in r.genwalk() if isinstance(rr.ogt,GRule) and rr.ogt.rname=='oneline']

    #for rr in nesting_nodes:
    #    print rr

    np.random.seed(8)
    np.random.choice(nesting_nodes).inrand=None

    #for rr in r.genwalk():
    #    rr.inrand=None

    for randstate, s in islice(x.generate(r),3):
        print('----')
        print(s)

if __name__=='__main__':
    #resampletest()
    resampletest2()
    #gensamples()

# vim: ts=4 sw=4

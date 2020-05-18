#!/usr/bin/env python
'''

    a mathematically silly example that demonstrates resampling for optimization.

    we generate polynomials in a variable 'x' and measure its fit to a fixed polynomial.

'''
from __future__ import absolute_import, division, print_function

import random

from datetime import datetime,timedelta

from gramma import *

class FunctionGrammar(GrammaGrammar):
    G=r'''
        start := expr;

        expr := add;

        add :=  mul . '+' . mul | `min(.01,depth/30.0)` mul ;
        mul :=  atom . '*' . atom | `min(.01,depth/30.0)` atom ;

        atom :=  'x' | const() | `min(.01,depth/30.0)` "(" . expr . ")" ;

    '''

    def __init__(x):
        GrammaGrammar.__init__(x,type(x).G, sideeffects=[DepthTracker])

    @gfunc
    def const(x):
        yield '%f' %(10*(x.random.rand()-.5))
        #yield str(x.random.choice([1,2,3,4]))


g=FunctionGrammar()
sampler=GrammaSampler(g)
tracer=Tracer()
sampler.add_sideeffects(tracer)

inputs=np.arange(-10,10,.5)
#outputs0=4+3*inputs+2*inputs**2+1*inputs**3
#outputs0=1+2*inputs+3*inputs**2+4*inputs**3
outputs0=1+inputs+inputs**2+inputs**3
#outputs0=inputs**2+inputs**4

def samp(ge=None):
    e=sampler.sample(ge)
    f=lambda x:eval(e)
    outputs=np.array([f(x) for x in inputs])
    #score=np.linalg.norm(outputs-outputs0)
    score=np.abs(outputs-outputs0).max()
    return e,score


factor=10
def meth0():
    'wild west'
    return samp()

def meth1():
    'fiddle with alts of last best'
    if tt==None:
        return None
    rge,cfg=tt.resample_mostly(g,lambda t:False, factor=factor)
    sampler.update_cache(cfg)
    return samp(rge)

def mkmeth(pred):
    'resample nodes'
    def meth():
        if tt==None:
            return None
        nodes=[n for n in tt.gennodes() if pred(n)]
        if len(nodes)==0:
            return None
        n=random.choice(nodes)
        rge,cfg=tt.resample_mostly(g,lambda t:t==n, factor=factor)
        sampler.update_cache(cfg)
        return samp(rge)
    return meth


class Timeleft:
    def __init__(self, seconds):
        self.done=datetime.now()+timedelta(seconds=seconds)
    def __bool__(self):
        return datetime.now()<=self.done
    __nonzero__=__bool__
    def over(self):
        return datetime.now()-self.done

#meths=[meth0, mkmeth(lambda n:isinstance(n.ge,GRule)), mkmeth(lambda n:isinstance(n.ge,GFunc))]
#meths=[meth0, mkmeth(lambda n:isinstance(n.ge,GRule))]
meths=[meth0, meth1]

try:
    while True:
        tl=Timeleft(10)
        beste=None
        tt=None
        bestscore=1e10
        #p=np.random.random(len(meths))
        p=np.array([.5,.5])
        #a=5
        #factor=np.random.pareto(a)*(a-1)*10.  # mean lomax(a) = 1/(a-1), var = a/((a-1)(a-1)(a-2)).. so larger a give tighter variance
        factor=10

        p/=p.sum()

        while tl:
            mi=np.random.choice(range(len(meths)),p=p)
            meth=meths[mi]
            escore=meth()
            if escore==None:
                continue
            e,score=escore

            if score < bestscore:
                beste,bestscore,tt=e,score,tracer.tracetree

                f=lambda x:eval(e)
                outputs=np.array([f(x) for x in inputs])
                errs=np.abs(outputs-outputs0)

                print('%d %8.2f %8.2f %8.2f: %s' % (mi, bestscore, errs.mean(), errs.std(), beste))
        if tl.over().total_seconds()<.1:
            print('   %8.2f %8.2f %s' % (bestscore, factor, ''.join('%8.3f' % x for x in  p)))
        sys.stdout.flush()
except KeyboardInterrupt:
    print('\nctrl-c pressed')


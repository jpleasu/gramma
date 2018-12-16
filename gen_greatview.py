#!/usr/bin/env python
import pdb

from gramma import *

class Greatview(Gramma):
    def __init__(x):
        with open('gv2.glf') as infile:
            Gramma.__init__(x,infile.read())
        x.remembered={}

    def rand_val(x):
        x.exp=random.choice([8,16,32])
        x.lastval=random.randrange(0,2**x.exp-1)
        return '%d' % x.lastval

    def bigger_val(x):
        v=random.randrange(x.lastval+1,2**x.exp)
        return '%d' % v

    def old(x,namespace):
        namespace=x.getstring(namespace)
        return random.choice(list(x.remembered.get(namespace)))
        
    def new(x,namespace,child):
        namespace=x.getstring(namespace)
        names=x.remembered.setdefault(namespace,set())
        n=x.sample(child)
        while n in names:
            n=x.sample(child)
        names.add(n)
        return n
            
    def reset(x):
        x.lastval=None
        x.exp=None
        x.remembered.clear()

def run():
    g=Greatview()
    
    #np.random.seed(1)
    #random.seed(1)
    for x in g.generate():
        try:
            print('----')
            print(x)
            #print(len(x))
        except KeyboardInterrupt:
            break
        except:
            pass


#pdb.run('run()')
run()


# vim: ts=4 sw=4

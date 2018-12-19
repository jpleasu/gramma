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

def run():
    g=Greatview()
    
    np.random.seed(1)
    for x in g.generate():
        try:
            if len(x)>0:
                print('----')
                print(x)
        except KeyboardInterrupt:
            break
        except:
            pass


#pdb.run('run()')
run()


# vim: ts=4 sw=4

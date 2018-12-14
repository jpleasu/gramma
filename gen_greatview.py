#!/usr/bin/env python

from gramma import *

class Greatview(Gramma):
    def __init__(x):
        Gramma.__init__(x,r'''
            start := 
                def . (def{0,1.} . use{1,3.}) {1,3.}
                . ineq{1,3.}
            ;

            def :=    "newthing " . new('thing',id) . ";\n" ;
            use :=    "oldthing " . old('thing') . ";\n" ;

            ineq := rand_val() . " < " . bigger_val() . ";\n" ;

            id := ['a'..'z'] . ['a'..'z']{5.};

        ''')
        # x.remembered[namespace]=set(names)
        x.remembered={}

    def rand_val(x):
        while True:
            x.exp=random.choice([8,16,32])
            x.lastval=random.randrange(0,2**x.exp-1)
            yield '%d' % x.lastval

    def bigger_val(x):
        while True:
            v=random.randrange(x.lastval+1,2**x.exp)
            yield '%d' % v

    def old(x,namespace):
        namespace=x.getstring(namespace)
        while True:
            #print('old yield %s' % x.remembered)
            yield random.choice(list(x.remembered.get(namespace)))
        
    def new(x,namespace,child):
        namespace=x.getstring(namespace)
        it=x.build(child)
        while True:
            names=x.remembered.setdefault(namespace,set())
            n=it.next()
            while n in names:
                n=it.next()
            names.add(n)
            #print('new yield %s' % x.remembered)
            yield n
            
    def reset(x):
        x.remembered.clear()

g=Greatview()
it=g.generate()
for i in xrange(10):
    print('---')
    print(it.next())


# vim: ts=4 sw=4

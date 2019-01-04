#!/usr/bin/env python

from gramma import *

class Example(Gramma):
    g=r'''
        start0 := ab . cd . ef;

        start := start0;

        ab := ['a' .. 'b'];
        cd := ['c' .. 'd']{1,3};
        #ef := rando(['e' .. 'f']);
        ef := randchar();

    '''
    @gfunc(stateful=False)
    def rando(x,arg):
        return '|%s|' % x.sample(arg)

    @gfunc(stateful=False)
    def randchar(x):
        return 'f'

    def __init__(x):
        Gramma.__init__(x,Example.g)

if __name__=='__main__':
    x=Example()
    x.random.seed(4)
    rseed=x.random.get_state()

    print('== rules ==')
    for rname,rt in sorted(x.ruledefs.iteritems()):
        print('%s = %s' % (rname,rt))

    r=x.build_richsample(rseed)
    print('== r, the richsample ==')
    print('%s' % r.to_gtree(x))
    print('-- r.s --')
    print('%s' % r.s)
 
    print('-- rule node to resample --')
    n=r.gen_rule_nodes('cd').next()
    print(n)
    n.inrand=None

    print('== r, the richsample, after resampling n ==')
    print('%s' % r.to_gtree(x))

    print('-- samples --')
    x.random.seed(5)
    for rand, s in islice(x.generate(r),5):
        print(s)
        print('~~~~')

# vim: ts=4 sw=4

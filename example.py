#!/usr/bin/env python

from gramma2 import *

class Example(GrammaGrammar):
    G=r'''
        start := recurs;

        recurs := 10 ".".recurs | words . " " . ['1'..'9'] . digit{1,15,geom(5)};

        digit := ['0' .. '9'];

        words := ( .25 "stink" | .75 "stank" )." ".f();
    '''

    ALLOWED_IDS=['g_allowed']

    def __init__(x):
        GrammaGrammar.__init__(x,type(x).G)

    @gfunc
    def f(self):
        return self.random.choice(['woof','meow'])

    @gfunc()
    def g(self):
        return 'g_return' + g_allowed

    @gfunc(noauto=True)
    def gg(self):
        return 'gg_return' + g_not_allowed

    @gfunc(statevars=['extrastate'])
    def h(self):
        return 'h_return'


def test_samples():
    g=Example()
    it=g.generate()
    for i in xrange(10):
        print(it.next()[1])

def test_parse():
    global g
    g=Example()
    print(g.parse('''"a"|"b"'''))

def test_constraining():

    class LeftAlt(Sampler):
        '''
            a sampler that forces every Alt to take the left option up to
            maximum expression depth
        '''
        def __init__(self,base,maxdepth=5):
            Sampler.__init__(self,base)
            _=type('_',(),{})
            object.__setattr__(self,'_',_)
            _.maxdepth=maxdepth
            _.d=0

        def sample(self,ge):
            _=self._
            try:
                _.d+=1
                if isinstance(ge,GAlt):
                    if _.d<=_.maxdepth:
                        return ge.children[0].do_sample(self)
                return ge.do_sample(self)
            finally:
                _.d-=1

    class C(Sampler):
        __slots__='stack',
        def __init__(self,base):
            Sampler.__init__(self,base)
            object.__setattr__(self,'stack',[])

        def sample(self,ge):
            self.stack.append(ge)
            print(ge)
            s=ge.do_sample(self)
            self.stack.pop()
            return s

    g=Example()
    #sampler=C(g)
    sampler=LeftAlt(g,10)
    for i in range(10):
        #print(sampler.sample(g.parse('start')))
        print(sampler.sample(g.ruledefs['start']))

if __name__=='__main__':
    #test_samples()
    #test_parse()
    test_constraining()


# vim: ts=4 sw=4

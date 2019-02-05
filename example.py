#!/usr/bin/env python

from gramma2 import *
from builtins import super

class Example(GrammaGrammar):
    G=r'''
        start := recurs;

        recurs := 10 ".".recurs | words . " " . ['1'..'9'] . digit{1,15,geom(5)};

        digit := ['0' .. '9'];

        words := ( .25 "stink" | .75 "stank" ).(" f=".f()." ff=".ff()){1,4};
    '''

    ALLOWED_IDS=['g_allowed']

    def __init__(x):
        GrammaGrammar.__init__(x,type(x).G)

    def reset_state(self,state):
        state.a=0
        state.x=type('_',(),{})
        state.x.y=7
        state.extrastate={}

    @gfunc
    def f(x):
        return x.random.choice(['woof','meow'])

    @gfunc
    def ff(x):
        x.state.a^=1
        return ['bleep','bloop'][x.state.a]


    @gfunc()
    def g(x):
        return 'g_return' + g_allowed

    @gfunc(noauto=True)
    def gg(x):
        return 'gg_return' + g_not_allowed

    @gfunc(statevars=['extrastate'])
    def h(x):
        return 'h_return'

    @gfunc
    def hh(x):
        x.state.extrastate=7
        return ''


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

    class LeftAlt(GrammaSampler):
        '''
            a sampler that forces every Alt to take the left option up to
            maximum expression depth
        '''
        def __init__(self,grammar,maxdepth=5):
            GrammaSampler.__init__(self,grammar)
            self.maxdepth=maxdepth
            self.d=0

        def sample(self,ge):
            self.d+=1
            try:
                if isinstance(ge,GAlt):
                    if self.d<=self.maxdepth:
                        return super().sample(ge.children[0])
                return super().sample(ge)
            finally:
                self.d-=1

    class C(GrammaSampler):
        __slots__='stack',
        def __init__(self,base):
            GrammaSampler.__init__(self,base)
            object.__setattr__(self,'stack',[])

        def sample(self,ge):
            self.stack.append(ge)
            print(ge)
            s=super().sample(ge)
            self.stack.pop()
            return s

    g=Example()
    #sampler=GrammaSampler(g)
    sampler=LeftAlt(g,50)
    #sampler=C(g)

    for i in range(10):
        #print(sampler.sample(g.parse('start')))
        sampler.reset()
        print(sampler.sample(g.ruledefs['start']))

if __name__=='__main__':
    #test_samples()
    #test_parse()
    test_constraining()


# vim: ts=4 sw=4

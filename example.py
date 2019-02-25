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
        super().reset_state(state)
        state.a=0
        state.x=type('_',(),{})
        state.x.y=7
        state.extrastate={}

    @gfunc
    def f(x):
        yield x.random.choice(['woof','meow'])

    @gfunc
    def ff(x):
        x.state.a ^= 1
        yield ['bleep','bloop'][x.state.a]


    @gfunc()
    def g(x):
        yield 'g_return' + g_allowed

    @gfunc(noauto=True)
    def gg(x):
        yield 'gg_return' + g_not_allowed

    @gfunc(statevars=['extrastate'])
    def h(x):
        yield 'h_return'

    @gfunc
    def hh(x):
        x.state.extrastate=7
        yield ''


def test_samples():
    g=Example()
    it=g.generate()
    for i in xrange(10):
        print(it.next())

def test_parse():
    global g
    g=Example()
    print(g.parse('''"a"|"b"'''))

def demo_coroutine_versus_recursive():
    '''
        Two ways to create a sampler that forces every Alt to take the left
        option up to maximum expression depth.  The second method which uses
        recursion is arguably simpler, but less composable.
    '''
    g=Example()

    print('==================')
    class LeftAlt(GrammaSampler):
        def __init__(self,grammar,maxdepth=5):
            GrammaSampler.__init__(self,grammar)
            self.maxdepth=maxdepth
            self.d=0

        def reset(self):
            super().reset()
            self.d=0

        def extract(self,resp):
            return super().extract(resp)[0]

        def recv(self,req):
            if isinstance(req,GExpr):
                if isinstance(req,GAlt):
                    self.d+=1
                    if self.d<=self.maxdepth:
                        # to handle nested alts, we must push twice
                        def shimmy(x):
                            yield (yield req.children[0])
                        return (shimmy(self.x),True)
                    else:
                        return (super().recv(req),True)
            else: # it's a string from the current top
                if self.stack[-1][1]:
                    self.d-=1
            return (super().recv(req),False)


    sampler=LeftAlt(g,50)
    sampler.random.seed(0)

    for i in range(10):
        s=sampler.sample()
        print('%3d %s' % (len(s),s))


    print('==================')
    class LeftAltR(GrammaSampler):
        def __init__(self,grammar,maxdepth=5):
            GrammaSampler.__init__(self,grammar)
            self.maxdepth=maxdepth
            self.d=0

        def reset(self):
            super().reset()
            self.d=0

        def rsample(self,ge):
            if isinstance(ge,GAlt):
                self.d+=1
                if self.d<=self.maxdepth:
                    res=self.rsample(ge.children[0])
                else:
                    res=super().rsample(ge)
                self.d-=1
                return res
            return super().rsample(ge)

    sampler=LeftAltR(g,50)
    sampler.random.seed(0)

    for i in range(10):
        sampler.reset()
        s=sampler.rsample(g.parse('start'))
        print('%3d %s' % (len(s),s))


def atleast(n,it):
    c=0
    for _ in it:
        c+=1
        if c>=n:
            return True
    return False
def itlen(it):
    c=0
    for _ in it:
        c+=1
    return c

def demo_tracetree():
    '''
        generate a tracetree for a sample then pick an alt node to re-sample with chosen bias.
    '''
    from collections import Counter

    g=Example()
    sampler=TracingSampler(g)
    sampler.random.seed(0)

    for i in range(10):
        s=sampler.sample()
        print('---------------')
        print('%3d %s' % (len(s),s))
        #print(sampler.tt)
        sampler.tracetree.dump()



def test_rlim():
    g=GrammaGrammar('start :=  rlim("a" . start, 3, "");')
    sampler=GrammaSampler(g)
    sampler.random.seed(0)
    for i in range(10):
        s=sampler.sample()
        print('---------------')
        print('%3d %s' % (len(s),s))


if __name__=='__main__':
    #test_parse()
    #test_samples()
    #demo_coroutine_versus_recursive()
    #demo_tracetree()
    test_rlim()


# vim: ts=4 sw=4

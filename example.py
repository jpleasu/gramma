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


def test_parse():
    global g
    g=Example()
    print(g.parse('''"a"|"b"'''))


def test_samples():
    g=Example()
    it=g.generate()
    for i in xrange(10):
        print(next(it))


def test_rlim():
    g=GrammaGrammar('start :=  rlim("a" . start, 3, "");')
    ctx=SamplerContext(g)
    ctx.random.seed(0)
    sampler=DefaultSampler(ctx)
    for i in range(10):
        s=ctx.sample(sampler)
        print('---------------')
        print('%3d %s' % (len(s),s))


class LeftAltSampler(object):
    def __init__(self,base,ctx,maxdepth=5):
        self.base=base
        self.maxdepth=maxdepth
        self.ctx=ctx
        self.d=0

    def reset(self):
        self.base.reset()
        self.d=0

    def extract(self,resp):
        return self.base.extract(resp)[0]

    def recv(self,req):
        if isinstance(req,GExpr):
            if isinstance(req,GAlt):
                self.d+=1
                if self.d<=self.maxdepth:
                    # to handle nested alts, we must push twice
                    def shimmy(x):
                        yield (yield req.children[0])
                    return (shimmy(self.ctx.x),True)
                else:
                    return (self.base.recv(req),True)
        else: # it's a string from the current top
            if self.ctx.stack[-1][1]:
                self.d-=1
        return (self.base.recv(req),False)




def test_constraint():
    '''
        A sampler that forces every Alt to take the left option up to maximum
        expression depth.
    '''
    g=Example()

    print('==================')

    ctx=SamplerContext(g)
    sampler=LeftAltSampler(DefaultSampler(ctx),ctx,50)
    ctx.random.seed(0)

    for i in range(10):
        s=ctx.sample(sampler)
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

def test_tracetree():
    '''
        generate a tracetree for a sample then pick an alt node to re-sample with chosen bias.
    '''
    g=Example()
    ctx=SamplerContext(g)
    ctx.random.seed(0)
    sampler=TracingSampler(DefaultSampler(ctx))

    for i in range(10):
        s=ctx.sample(sampler)
        print('---------------')
        print('%3d %s' % (len(s),s))
        #print(sampler.tt)
        sampler.tracetree.dump()

def test_composition():
    g=Example()
    ctx=SamplerContext(g)
    ctx.random.seed(0)
    sampler=TracingSampler(LeftAltSampler(DefaultSampler(ctx),ctx,50))
    for i in range(10):
        s=ctx.sample(sampler)
        print('---------------')
        print('%3d %s' % (len(s),s))
        #print(sampler.tt)
        sampler.tracetree.dump()


def test_resample():
    g=Example()
    ctx=SamplerContext(g)
    ctx.random.seed(0)
    sampler=DefaultSampler(ctx)

    def again():
        s=ctx.sample(sampler)
        print('---------------')
        print('%3d %s' % (len(s),s))
    again()
    r0=ctx.state.randstates['__initial_random']
    again()

    class SamplerContext2(SamplerContext):
        __slots__='randstates',

        def reset(self):
            super().reset()
            self.state.randstates.update(self.randstates)

    ctx2=SamplerContext2(g)
    sampler2=DefaultSampler(ctx2)
    ctx2.randstates=dict(r0=r0)
    s=ctx2.sample(sampler2, 'set_rand(r0).start')
    print('---------------')
    print('%3d %s' % (len(s),s))



if __name__=='__main__':
    #test_parse()
    #test_samples()
    #test_rlim()
    #test_constraint()
    #test_tracetree()
    #test_composition()
    test_resample()


# vim: ts=4 sw=4

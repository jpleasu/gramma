#!/usr/bin/env python

from gramma2 import *
from builtins import super

class Example(GrammaGrammar):
    G=r'''
        start := recurs;

        recurs := 10 ".".recurs | words . " " . ['1'..'9'] . digit{1,15,geom(5)};

        digit := ['0' .. '9'];

        words := ( .25 "dog" | .75 "cat" ).(" f=".f()." ff=".ff()){1,4};
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


def demo_parser():
    global g
    g=Example()
    print(g.parse('''"a"|"b"'''))


def demo_sampling():
    g=Example()
    it=g.generate()
    for i in xrange(10):
        print(next(it))


def demo_rlim():
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
            if len(self.ctx.stack)>0 and self.ctx.stack[-1][1]:
                self.d-=1
        return (self.base.recv(req),False)

def demo_constraint():
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

def demo_tracetree():
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

def demo_composition():
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


def demo_resample():
    g=Example()
    ctx=SamplerContext(g)
    ctx.random.seed(0)
    sampler=DefaultSampler(ctx)

    def p(n, e):
        s=ctx.sample(sampler,e)
        print('---- %s ----' % n)
        print('%3d %s' % (len(s),s))
    # generate a sample and save the random state on entry
    p('A', 'save_rand(r0).start')
    # generate a new sample, demonstrating a different random state
    p('B', 'start')

    # resume at r0 again:
    p('A', 'load_rand(r0).start')
    # if we generate a new sample here, the state resumes from r0 again, so it will be A
    p('B', 'start')

    # so if we want to resume at r0..
    p('A', 'load_rand(r0).start')
    # .. and continue w/ a new random, we need to reseed:
    ctx.random.seed(None) # none draws a new seed from urandom
    p('C', 'start')

    # and we can still resume from r0 later.
    p('A', 'load_rand(r0).start')
    # we can also reseed the random number generator from within the grammar
    p('D', 'reseed_rand().start')



if __name__=='__main__':
    #demo_parser()
    #demo_sampling()
    #demo_rlim()
    demo_constraint()
    #demo_tracetree()
    #demo_composition()
    #demo_resample()

# vim: ts=4 sw=4

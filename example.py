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
    sampler=DefaultSampler(g)
    for i in range(10):
        s=ctx.sample(sampler)
        print('---------------')
        print('%3d %s' % (len(s),s))


class LeftAltSampler(ProxySampler):
    def __init__(self,base,maxdepth=5):
        ProxySampler.__init__(self,base)
        self.maxdepth=maxdepth
        self.d=0

    def reset(self):
        self.base.reset()
        self.d=0

    def unwrap(self,top):
        # all but "id_shim" stack objects are of the form (base wrapped gen, boolean)
        # "id_shim" is (id_shim(x), True)
        # so.. don't call base unwrap on shimmys?
        return self.base.unwrap(top[0])

    def complete(self,top,s):
        self.base.complete(top[0],s)
        if top[1]:
            self.d-=1

    def wrap(self,ge,b):
        ctor=self.base.expr2ctor(ge)
        return lambda x:(ctor(x),b)

    def expr2ctor(self,ge):
        if isinstance(ge,GAlt):
            self.d+=1
            if self.d<=self.maxdepth:
                # handle nested alts
                def id_shim(x):
                    yield (yield ge.children[0])
                return lambda x:(id_shim(x),True)
            else:
                return self.wrap(ge,True)
        return self.wrap(ge,False)

def demo_constraint():
    '''
        A sampler that forces every Alt to take the left option up to maximum
        expression depth.
    '''
    g=Example()

    print('==================')

    ctx=SamplerContext(g)
    sampler=LeftAltSampler(DefaultSampler(g),50)
    ctx.random.seed(0)

    for i in range(10):
        s=ctx.sample(sampler)
        print('%3d %s' % (len(s),s))
        assert(ctx.state.d==0)

def demo_tracetree():
    '''
        generate a tracetree for a sample then pick an alt node to re-sample with chosen bias.
    '''
    g=Example()
    ctx=SamplerContext(g)
    ctx.random.seed(0)
    sampler=TracingSampler(DefaultSampler(g),ctx.random)

    for i in range(10):
        s=ctx.sample(sampler)
        print('---------------')
        print('%3d %s' % (len(s),s))
        sampler.tracetree.dump()

def demo_composition():
    g=Example()
    ctx=SamplerContext(g)
    ctx.random.seed(0)
    sampler=TracingSampler(LeftAltSampler(DefaultSampler(g),5),ctx.random)
    for i in range(10):
        s=ctx.sample(sampler)
        print('---------------')
        print('%3d %s' % (len(s),s))
        sampler.tracetree.dump()


def demo_random_states():
    g=Example()
    ctx=SamplerContext(g)
    ctx.random.seed(0)
    sampler=DefaultSampler(g)

    def p(n, e):
        s=ctx.sample(sampler,e)
        print('---- %s ----' % n)
        print('%3d %s' % (len(s),s))

    # generate a sample and save the random state on entry
    p('A', 'save_rand("r0").start')
    # generate a new sample, demonstrating a different random state
    p('B', 'start')

    # resume at r0 again:
    p('A', 'load_rand("r0").start')
    # if we generate a new sample here, the state resumes from r0 again, so it will be A
    p('B', 'start')

    # so if we want to resume at r0..
    p('A', 'load_rand("r0").start')
    # .. and continue w/ a new random, we need to reseed:
    ctx.random.seed(None) # none draws a new seed from urandom
    p('C', 'start')

    # and we can still resume from r0 later.
    p('A', 'load_rand("r0").start')
    # we can also reseed the random number generator from within the grammar
    p('D', 'reseed_rand().start')


def demo_resample():
    import random

    g=Example()
    ctx=SamplerContext(g)
    #ctx.random.seed(4)
    ctx.random.seed(2)
    sampler0=DefaultSampler(g)
    sampler=TracingSampler(sampler0,ctx.random)
    print(ctx.sample(sampler))

    tt=sampler.tracetree

    # resample the same thing with a cached randstate:
    ctx.random.set_cached_state('r',tt.inrand)
    print(ctx.sample(sampler0,'load_rand("r").start'))


    # same thing, but unwind to a node, reseed on enter, and reset to outrand
    # on exit.
    def gennodes(t):
        yield t
        for c in t.children:
            for tc in gennodes(c):
                yield tc

    def depth(t,d=0):
        if t.parent==None:
            return d
        return depth(t.parent,d+1)

    allnodes=list(gennodes(tt))
    #random.seed(5)
    #n=random.choice([n for n in allnodes if isinstance(n.ge,GRule)])
    #n=random.choice([n for n in allnodes if isinstance(n.ge,GAlt)])
    n=random.choice([n for n in allnodes if isinstance(n.ge,GRange)])
    print(depth(n))
    #n.dump()

    def resample(ge):
        return GCat([g.parse('reseed_rand()'),ge,g.parse('load_rand("r1")')])
        #return ge
    ctx.random.set_cached_state('r1',n.outrand)

    def t2ge(t):
        ge=t.ge
        if t==n:
            return resample(ge)
        elif isinstance(ge,GAlt):
            # XXX set rand according to first child, o/w stream is off?
            return t2ge(t.children[0])
        elif isinstance(ge,GRule):
            return t2ge(t.children[0])
        elif isinstance(ge,GCat):
            return GCat([t2ge(c) for c in t.children])
        elif isinstance(ge,GRep):
            # XXX set rand according to first child, o/w stream is off?
            return GCat([t2ge(c) for c in t.children])
        elif isinstance(ge,GRange):
            return GTok.from_str(t.s)
        elif isinstance(ge,GTok):
            return ge.copy()
        elif isinstance(ge,GFunc):
            return ge.copy() # don't recurse into children, since bare might be string arguments.. how can we tell? maybe all bare names whould be interpreted as grammatical
        else:
            raise ValueError('unknown GExpr node type: %s' % type(ge))
    rge=t2ge(tt)
    #print(rge)
    rge=rge.simplify()
    print(rge)
    print(ctx.sample(sampler0,rge))

if __name__=='__main__':
    #demo_parser()
    #demo_sampling()
    #demo_rlim()
    #demo_constraint()
    #demo_tracetree()
    #demo_composition()
    #demo_random_states()
    demo_resample()

# vim: ts=4 sw=4

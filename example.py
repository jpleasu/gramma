#!/usr/bin/env python

from gramma2 import *
from builtins import super

class BasicGrammar(GrammaGrammar):
    G=r'''
        start := recurs;

        recurs := 10 ".".recurs | words . " " . ['1'..'9'] . digit{1,15,geom(5)};

        digit := ['0' .. '9'];

        words := (`1000*(depth>20)` "*" | " ").( .25 "dog" | .75 "cat" ).(" f=".f()." ff=".ff()){1,4};
    '''

    ALLOWED_GLOBAL_IDS=['g_allowed']

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

    @gfunc(statevar_defs=['extrastate'])
    def h(x):
        yield 'h_return'

    @gfunc
    def hh(x):
        x.state.extrastate=7
        yield ''


class ArithmeticGrammar(GrammaGrammar):

    G=r'''
        start := expr;

        expr := add;

        add :=  mul . '+' . mul | `depth/30.0` mul ;
        mul :=  atom . '*' . atom | `depth/30.0` atom ;

        atom :=  var | 3 int | "(" . expr . ")";

        var := ['a'..'z']{1,5,geom(3)} ;
        int := ['1'..'9'] . digit{1,8,geom(3)};

        digit := ['0' .. '9'];
    '''

    def __init__(x):
        GrammaGrammar.__init__(x,type(x).G)





def demo_parser():
    g=BasicGrammar()
    print(g.parse('''
        "a"| `cats` "b"
    '''))


def demo_sampling():
    g=BasicGrammar()
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
    g=BasicGrammar()

    print('==================')

    ctx=SamplerContext(g)
    sampler=LeftAltSampler(DefaultSampler(g),50)
    ctx.random.seed(0)

    for i in range(10):
        s=ctx.sample(sampler)
        print('%3d %s' % (len(s),s))
        assert(sampler.d==0)

def demo_tracetree():
    '''
        generate a tracetree for a sample then pick an alt node to re-sample with chosen bias.
    '''
    g=BasicGrammar()
    ctx=SamplerContext(g)
    ctx.random.seed(0)
    sampler=TracingSampler(DefaultSampler(g),ctx.random)

    for i in range(10):
        s=ctx.sample(sampler)
        print('---------------')
        print('%3d %s' % (len(s),s))
        sampler.tracetree.dump()

def demo_composition():
    g=BasicGrammar()
    ctx=SamplerContext(g)
    ctx.random.seed(0)
    sampler=TracingSampler(LeftAltSampler(DefaultSampler(g),5),ctx.random)
    for i in range(10):
        s=ctx.sample(sampler)
        print('---------------')
        print('%3d %s' % (len(s),s))
        sampler.tracetree.dump()


def demo_random_states():
    g=BasicGrammar()
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

    g=ArithmeticGrammar()
    ctx=SamplerContext(g)
    #ctx.random.seed(15)
    sampler0=DefaultSampler(g)
    sampler=TracingSampler(sampler0,ctx.random)
    origs=ctx.sample(sampler)
    print('-- the original sample --')
    print(origs)

    tt=sampler.tracetree
    # resample with the cached randstate.
    # Note that the dynamic alternations used in ArithmeticGrammar use depth
    # and using "cat(load_rand,start)" increases the depth. Reset the depth
    # with the 'def'. (depth is set to 1 because on _exit_ from the gfunc
    # call, depth is decremented)
    ctx.random.set_cached_state('r',tt.inrand)
    s=ctx.sample(sampler0,'load_rand("r").def("depth",1).start')
    assert(s==origs)

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
        if t.ge.is_rule('expr'):
            return depth(t.parent,d+1)
        return depth(t.parent,d)

    allnodes=list(gennodes(tt))
    #random.seed(5)
    n=random.choice([n for n in allnodes if isinstance(n.ge,GRule)])
    #n=random.choice([n for n in allnodes if isinstance(n.ge,GAlt)])
    #n=random.choice([n for n in allnodes if isinstance(n.ge,GRange)])
    print('depth(n) = %d' % depth(n))
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
            # XXX lost sample, rand stream out of sync w/ original
            return t2ge(t.children[0])
        elif isinstance(ge,GRule):
            return t2ge(t.children[0])
        elif isinstance(ge,GCat):
            return GCat([t2ge(c) for c in t.children])
        elif isinstance(ge,GRep):
            # XXX lost sample, rand stream out of sync w/ original
            return GCat([t2ge(c) for c in t.children])
        elif isinstance(ge,GRange):
            return GTok.from_str(t.s)
        elif isinstance(ge,GTok):
            return ge.copy()
        elif isinstance(ge,GFunc):
            # XXX: don't recurse into children, since bare might be string
            # arguments.. how can we tell? maybe all bare names whould be
            # interpreted as grammatical
            return ge.copy()
        else:
            raise ValueError('unknown GExpr node type: %s' % type(ge))
    rge=t2ge(tt)
    #print(rge)
    rge=rge.simplify()
    print('-- the resample expression --')
    print(rge)
    for i in range(10):
        print('---')
        print(ctx.sample(sampler0,rge))


def demo_resample2():
    import random
    g=ArithmeticGrammar()
    ctx=SamplerContext(g)
    sampler0=DefaultSampler(g)
    sampler=TracingSampler(sampler0,ctx.random)

    s=ctx.sample(sampler)
    print(s)
    i=random.randrange(0,len(s))
    print(' '*i + '''^- looking for this char's origin''')
    tt=sampler.tracetree
    n=tt.child_containing(i)
    d=0
    while n!=None:
        print('   %s%s' % (' '*d, n.ge))
        n=n.parent
        d+=1
    #n.dump()
    #print(n.ge)


 
def demo_grammar_analysis():

    # if a state variable is used by a gfunc, it must be reset
    s=None
    try:
        class AnalyzeMeGrammar1(GrammaGrammar):
            @gfunc
            def f(x):
                yield str(x.state.m)

    except Exception as e:
        s=str(e)
    assert('has no reset_state method, but uses statespace(s) m' in s)

    s=None
    try:
        class AnalyzeMeGrammar1fix(GrammaGrammar):

            def reset_state(self,state):
                state.m=5

            @gfunc
            def f(x):
                yield str(x.state.m)

    except Exception as e:
        s=str(e)
    assert(s==None)


    # global are forbidden..
    s=None
    try:
        class AnalyzeMeGrammar2(GrammaGrammar):

            @gfunc
            def f(x):
                yield str(g_global)

    except Exception as e:
        s=str(e)
    assert("forbidden use of variable 'g_global' in f" in s)

    # unless explicitly allowed
    s=None
    try:
        class AnalyzeMeGrammar2fix(GrammaGrammar):

            ALLOWED_GLOBAL_IDS=['g_global']

            @gfunc
            def f(x):
                yield str(g_global)

    except Exception as e:
        s=str(e)
    assert(s==None)




    # gfuncs don't return their value, they yield it.
    s=None
    try:
        class AnalyzeMeGrammar3(GrammaGrammar):

            @gfunc
            def f(x):
                return 'my value'

    except Exception as e:
        s=str(e)
    assert("gfunc f of class AnalyzeMeGrammar3 doesn't yield a value" in s)

    s=None
    try:
        class AnalyzeMeGrammar3fix(GrammaGrammar):

            @gfunc
            def f(x):
                yield 'my value'

    except Exception as e:
        s=str(e)
    assert(s==None)






    s=None
    try:
        class AnalyzeMeGrammar4(GrammaGrammar):
            def reset_state(self,state):
                state.assigned=1
                state.used=1
                state.mod=1
                state.subscript_use={}
                state.subscript_def={}
                state.subscript_mod={}
                state.subscript_mod2={}
                state.obj1={}
                state.obj2={}
                state.obj3={}

            @gfunc
            def f(x):
                # use
                if x.state.used:
                    yield ''
                # def
                x.state.assigned=True

                # both
                x.state.mod+=1

                # use
                if x.state.subscript_use[15]:
                    yield ''
                # def
                x.state.subscript_def[15]=True

                # both
                x.state.subscript_mod[15]+=1

                # w/out knowing object types, we can't say, so all of the
                # following are both
                x.state.obj1.method()
                x.state.obj2.field=1
                x.state.obj3.field.method()

                # both.. can't tell with objects
                x.state.subscript_mod2[15].method()


                s=yield "donkey"
                yield s

    except Exception as e:
        s=str(e)

    #print(','.join(sorted(AnalyzeMeGrammar4.f.statevar_uses)))
    #print(','.join(sorted(AnalyzeMeGrammar4.f.statevar_defs)))

    assert('mod,obj1,obj2,obj3,subscript_mod,subscript_mod2,subscript_use,used'==','.join(sorted(AnalyzeMeGrammar4.f.statevar_uses)))
    assert('assigned,mod,obj1,obj2,obj3,subscript_def,subscript_mod,subscript_mod2'==','.join(sorted(AnalyzeMeGrammar4.f.statevar_defs)))



    s=None
    try:
        class AnalyzeMeGrammar5(GrammaGrammar):

            @gfunc
            def f(x):
                yield ''.join([(yield 'e%d') for e in range(3)])

    except Exception as e:
        s=str(e)
    assert('yield in a generator expression or list comprehension, in gfunc f of class AnalyzeMeGrammar5 on line 5' in s)


if __name__=='__main__':
    #demo_parser()
    #demo_sampling()
    #demo_rlim()
    #demo_constraint()
    #demo_tracetree()
    #demo_composition()
    #demo_random_states()
    demo_resample()
    #demo_resample2()
    #demo_grammar_analysis()

# vim: ts=4 sw=4

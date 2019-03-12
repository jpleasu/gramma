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

        add :=  mul . '+' . mul | `min(.01,depth/30.0)` mul ;
        mul :=  atom . '*' . atom | `min(.01,depth/30.0)` atom ;

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
    #g=BasicGrammar()
    g=ArithmeticGrammar()
    sampler=GrammaSampler(g)
    sampler.add_sideeffects(DepthTracker)
    for i in xrange(10):
        print(sampler.sample())


def demo_recursion_limits():
    ruleDepthTracker=DepthTracker(lambda ge:isinstance(ge,GRule))

    #g=GrammaGrammar('start :=  `depth<=3` "a" . start | `depth>3` "";')
    g=GrammaGrammar('start :=  `depth<=3` ? "a" . start : "" ;')
    sampler=GrammaSampler(g)
    sampler.add_sideeffects(ruleDepthTracker)
    sampler.random.seed(0)
    for i in range(10):
        s=sampler.sample()
        print('---------------')
        print('%3d %s' % (len(s),s))


def demo_tracetree():
    '''
        generate a tracetree for a sample then pick an alt node to re-sample with chosen bias.
    '''
    #g=BasicGrammar()
    g=ArithmeticGrammar()
    sampler=GrammaSampler(g)
    sampler.random.seed(0)
    tracer=Tracer()
    sampler.add_sideeffects(DepthTracker, tracer)

    for i in range(10):
        s=sampler.sample()
        print('---------------')
        print('%3d %s' % (len(s),s))
        tracer.tracetree.dump()

class LeftAltTransformer(Transformer):
    '''
        force sampler down left choice to an altdepth of maxdepth
    '''
    __slots__='maxdepth',

    def __init__(self,maxdepth=10):
        self.maxdepth=maxdepth

    def transform(self, x, ge):
        if x.state.altdepth<=self.maxdepth:
            if isinstance(ge,GAlt):
                nge=GAlt([1],[ge.children[0].copy()])
                nge.parent=ge
                return nge
        return ge

    altDepthTracker=DepthTracker(varname='altdepth', pred=lambda ge:isinstance(ge,GAlt))

def demo_transform():
    g=BasicGrammar()
    sampler=GrammaSampler(g)
    sampler.random.seed(0)

    sampler.add_sideeffects(DepthTracker, LeftAltTransformer.altDepthTracker)
    sampler.add_transformers(LeftAltTransformer(maxdepth=30))

    for i in range(10):
        s=sampler.sample()
        print('%3d %s' % (len(s),s))


def demo_random_states():
    g=BasicGrammar()
    sampler=GrammaSampler(g)
    sampler.add_sideeffects(DepthTracker)
    #sampler.random.seed(0)

    def p(n, e):
        s=sampler.sample(e)
        print('---- %s ----' % n)
        print('%3d %s' % (len(s),s))
        return s

    # generate a sample and save the random state on entry
    a=p('A', 'save_rand("r0").start')
    # generate a new sample, demonstrating a different random state
    b=p('B', 'start')

    # resume at r0 again:
    a1=p('A', 'load_rand("r0").start')
    assert(a1==a)
    # if we generate a new sample here, the state resumes from r0 again, so it will be A
    b1=p('B', 'start')
    assert(b1==b)

    # so if we want to resume at r0..
    a1=p('A', 'load_rand("r0").start')
    assert(a1==a)
    # .. and continue w/ a new random, we need to reseed:
    sampler.random.seed(None) # none draws a new seed from urandom
    c=p('C', 'start')

    # and we can still resume from r0 later.
    a1=p('A', 'load_rand("r0").start')
    assert(a1==a)
    # we can also reseed the random number generator from within the grammar
    d=p('D', 'reseed_rand().start')


def demo_resample():
    import random

    g=ArithmeticGrammar()
    #g=BasicGrammar()
    sampler=GrammaSampler(g)
    #sampler.random.seed(15)
    tracer=Tracer()
    sampler.add_sideeffects(DepthTracker, tracer)
    origs=sampler.sample()
    print('-- the original sample --')
    print(origs)

    tt=tracer.tracetree
    # resample with the cached randstate.
    # Note that the dynamic alternations used in ArithmeticGrammar use depth
    # and using "cat(load_rand,start)" increases the depth. Reset the depth
    # with the 'def'. (depth is set to 1 because on _exit_ from the gfunc
    # call, depth is decremented)
    sampler.random.set_cached_state('r',tt.inrand)
    s=sampler.sample('load_rand("r").def("depth",1).start')
    assert(s==origs)


    ## choose the node to resample, n
    allnodes=list(tt.gennodes())
    #random.seed(5)
    n=random.choice([n for n in allnodes if isinstance(n.ge,GRule)])
    #n=random.choice([n for n in allnodes if isinstance(n.ge,GAlt)])
    #n=random.choice([n for n in allnodes if isinstance(n.ge,GRange)])

    print('depth(n) = %d' % n.depth())
    #n.dump()

    ## construct a GExpr to resamples n with
    rge,cfg=tt.resample(g,lambda t:t==n)
    print('-- the resample expression --')
    print(rge)

    sampler.update_cache(cfg)

    for i in range(10):
        print('---')
        print(sampler.sample(rge))


def demo_resample2():
    import random
    g=ArithmeticGrammar()
    sampler=GrammaSampler(g)
    tracer=Tracer()
    sampler.add_sideeffects(DepthTracker, tracer)

    s=sampler.sample()
    print(s)
    i=random.randrange(0,len(s))
    print(' '*i + '''^- looking for this char's origin''')
    tt=tracer.tracetree
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
    #demo_recursion_limits()
    #demo_tracetree()
    #demo_transform()
    #demo_random_states()
    #demo_resample()
    demo_resample2()
    #demo_grammar_analysis()

# vim: ts=4 sw=4

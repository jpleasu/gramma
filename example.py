#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

from gramma2 import *
from builtins import super
import random

class DemoGrammar(GrammaGrammar):
    G=r'''
        start := recurs;

        recurs := 10 ".".recurs | words . " " . ['1'..'9'] . digit{1,15,geom(5)};

        digit := ['0' .. '9'];

        words := (`1000*(depth>20)` "*" | " ").( .25 "dog" | .75 "cat" ).(" f=".f()." ff=".ff()){1,4};

    '''

    

    def __init__(self):
        GrammaGrammar.__init__(self, self.G, sideeffects=[DepthTracker],allowed_global_ids=['g_allowed'])

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

    @staticmethod
    def punt(ge):
        pass

    @gfunc(analyzer=punt)
    def gg(x):
        yield 'gg_return' + g_not_allowed

    @gfunc(statevar_defs=set(['extrastate']))
    def h(x):
        yield 'h_return'

    @gfunc
    def hh(x):
        x.state.extrastate=7
        yield ''

    @gfunc
    def incra(x):
        x.state.a+=1
        yield ''

    @gfunc
    def geta(x):
        yield '%d' % (x.state.a)


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
        GrammaGrammar.__init__(x,type(x).G, sideeffects=[DepthTracker])

class ScopingGrammar(GrammaGrammar):

    G=r'''
        start := "-----\n".expr;

        ind := "  "{`depth/5-2`};
        expr := `depth<20`?scoped:ind." //";
        scoped := ind."{\n".
                  ind." def ".new(var).";\n".
                    (expr."\n"){,3}.
                  ind." use ".old().";\n".
                  ind."}";

        var := ['a'..'z']{1,5,geom(3)} ;
    '''

    def __init__(x):
        GrammaGrammar.__init__(x,type(x).G, sideeffects=[DepthTracker])

    def reset_state(self,state):
        state.vars=set()

    @gfunc
    def new(x,ge):
        n=yield ge
        x.state.vars.add(n)
        yield n

    @gfunc
    def old(x):
        yield x.random.choice(list(x.state.vars))



def demo_parser():
    g=GrammaGrammar('start:="";')
    #g=DemoGrammar()
    print(g.parse(r'''
        ['a'..'z']
    '''))
    print(g.parse(r'''
        '\''."insinglequotes".'\''
    '''))
    print(g.parse('''
        "a"| `15` "b"
    '''))
    print(g.parse('''
        "a"{1,2}
    '''))
    print(g.parse('''
        "a"{,2}
    '''))
    print(g.parse('''
        "a"{2}
    '''))


def demo_sampling():
    #g=DemoGrammar()
    #g=ArithmeticGrammar()
    g=ScopingGrammar()
    sampler=GrammaSampler(g)
    for i in xrange(10):
        print(sampler.sample())

def demo_recursion_limits():
    ruleDepthTracker=DepthTracker(lambda ge:isinstance(ge,GRule))
    g=GrammaGrammar('start :=  `depth<=3` ? "a" . start : "" ;', sideeffects=[ruleDepthTracker])
    sampler=GrammaSampler(g)
    sampler.random.seed(0)
    for i in range(10):
        s=sampler.sample()
        print('---------------')
        print('%3d %s' % (len(s),s))


def demo_tracetree():
    '''
        generate a tracetree for a sample then pick an alt node to re-sample with chosen bias.
    '''
    #g=DemoGrammar()
    g=ArithmeticGrammar()
    sampler=GrammaSampler(g)
    sampler.random.seed(0)
    tracer=Tracer()
    sampler.add_sideeffects(tracer)

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
    g=DemoGrammar()
    sampler=GrammaSampler(g)
    sampler.random.seed(0)

    sampler.add_sideeffects(LeftAltTransformer.altDepthTracker)
    sampler.add_transformers(LeftAltTransformer(maxdepth=30))

    for i in range(10):
        s=sampler.sample()
        print('%3d %s' % (len(s),s))


def demo_random_states():
    g=DemoGrammar()
    sampler=GrammaSampler(g)
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

    ## resample with the cached randstate.

    g=ArithmeticGrammar()
    sampler=GrammaSampler(g)
    tracer=Tracer()
    sampler.add_sideeffects(tracer)
    origs=sampler.sample()

    tt=tracer.tracetree
    # Note that the dynamic alternations used in ArithmeticGrammar use depth
    # and using "cat(load_rand,start)" increases the depth. Reset the depth
    # with the 'def'. (depth is set to 1 because on _exit_ from the gfunc
    # call, depth is decremented)
    r=tt.first(lambda n:n.ge.get_meta().uses_random).inrand
    sampler.random.set_cached_state('r',r)
    s=sampler.sample('def("depth",1).load_rand("r").start')
    assert(s==origs)


def demo_resample():
    import random
    g=ArithmeticGrammar()
    sampler=GrammaSampler(g)
    tracer=Tracer()
    sampler.add_sideeffects(tracer)

    origs=sampler.sample()
    tt=tracer.tracetree
    print('-- the original sample --')
    print(origs)

    ## choose the node to resample, n
    n=random.choice([n for n in tt.gennodes() if isinstance(n.ge,GRule)])

    print('-- resampling --')
    print('"%s" at depth(n) = %d' % (n.ge,n.depth()))

    ## construct a GExpr that resamples only n
    rge,cfg=tt.resample(g,lambda t:t==n)
    print('-- the resample expression --')
    print(rge)

    sampler.update_cache(cfg)

    for i in range(10):
        print('---')
        print(sampler.sample(rge))

class FuncyGrammar(GrammaGrammar):
    G=r'''
        start := r;
        r :=  "s."
            | `1-revcalled` "rev(".rev(r).")"
            | decide("t"|"f","a.".r,"b.".r) 
            | dub("a"|"b")
            | "c." . r;
    '''

    def __init__(self):
        GrammaGrammar.__init__(self, type(self).G, sideeffects=[DepthTracker])

    def reset_state(self, state):
        state.revcalled=False

    @gfunc
    def rev(x,child):
        x.state.revcalled=True
        yield ''.join(reversed((yield child)))

    @gfunc
    def decide(x,child1,child2,child3):
        s1=yield child1
        if 't' in s1:
            yield (yield child2)
        yield (yield child3)

    @gfunc
    def dub(x,child):
        yield (yield child)+(yield child)

def demo_resample_funcy():
    import random
    g=FuncyGrammar()
    sampler=GrammaSampler(g)
    tracer=Tracer()
    sampler.add_sideeffects(tracer)

    while True:
        origs=sampler.sample()
        tt=tracer.tracetree
        funcs=[n for n in tt.gennodes() if isinstance(n.ge,GFunc)]
        if len(funcs)>0:
            break
    f=random.choice(funcs)
    n=random.choice([n for n in f.gennodes() if not isinstance(n.ge,GFunc) and n.ge.get_meta().uses_random])
    print('-- the original sample --')
    print(origs)
    print('  with %d func node(s)' % len(funcs))
    print('chose func "%s" at depth(n) = %d' % (f.ge.fname, f.depth()))
    print(' and child %s at depth(n) = %d' % (n.ge,n.depth()))
    #tt.dump()

    ## construct a GExpr that resamples only n
    rge,cfg=tt.resample(g,lambda t:t==n)
    print('-- the resample expression --')
    print(rge)

    sampler.update_cache(cfg)

    for i in range(10):
        print('---')
        print(sampler.sample(rge))

from datetime import datetime,timedelta

class Timeleft:
    def __init__(self, seconds):
        self.done=datetime.now()+timedelta(seconds=seconds)
    def __nonzero__(self):
        return datetime.now()<=self.done
    def over(self):
        return datetime.now()-self.done
        
def demo_fit():
    '''
        fit a function w/ a sampled expression and resampling
    '''
    class FunctionGrammar(GrammaGrammar):
        G=r'''
            start := expr;
    
            expr := add;
    
            add :=  mul . '+' . mul | `min(.01,depth/30.0)` mul ;
            mul :=  atom . '*' . atom | `min(.01,depth/30.0)` atom ;
    
            atom :=  'x' | const() | `min(.01,depth/30.0)` "(" . expr . ")" ;
    
        '''
    
        def __init__(x):
            GrammaGrammar.__init__(x,type(x).G, sideeffects=[DepthTracker])

        @gfunc
        def const(x):
            yield '%f' %(10*(x.random.rand()-.5))


    g=FunctionGrammar()
    sampler=GrammaSampler(g)
    tracer=Tracer()
    sampler.add_sideeffects(tracer)

    inputs=np.arange(-10,10,.5)
    #outputs0=4+3*inputs+2*inputs**2+1*inputs**3
    outputs0=1+2*inputs+3*inputs**2+4*inputs**3
    #outputs0=1+1*inputs+1*inputs**2+1*inputs**3

    def samp(ge=None):
        e=sampler.sample(ge)
        f=lambda x:eval(e)
        outputs=np.array([f(x) for x in inputs])
        score=np.linalg.norm(outputs-outputs0)
        return e,score


    factor=10
    def meth0():
        'wild west'
        return samp()

    def meth1():
        'fiddle with alts of last best'
        if tt==None:
            return None
        rge,cfg=tt.resample_mostly(g,lambda t:False, factor=factor)
        sampler.update_cache(cfg)
        return samp(rge)

    def mkmeth(pred):
        'resample nodes'
        def meth():
            if tt==None:
                return None
            nodes=[n for n in tt.gennodes() if pred(n)]
            if len(nodes)==0:
                return None
            n=random.choice(nodes)
            rge,cfg=tt.resample_mostly(g,lambda t:t==n, factor=factor)
            sampler.update_cache(cfg)
            return samp(rge)
        return meth


    #meths=[meth0, mkmeth(lambda n:isinstance(n.ge,GRule)), mkmeth(lambda n:isinstance(n.ge,GFunc))]
    #meths=[meth0, mkmeth(lambda n:isinstance(n.ge,GRule))]
    meths=[meth0, meth1]

    try:
        while True:
            tl=Timeleft(10)
            beste=None
            tt=None
            bestscore=1e10
            #p=np.random.random(len(meths))
            p=np.array([.5,.5])
            #a=5
            #factor=np.random.pareto(a)*(a-1)*10.  # mean lomax(a) = 1/(a-1), var = a/((a-1)(a-1)(a-2)).. so larger a give tighter variance
            factor=10

            p/=p.sum()

            while tl:
                mi=np.random.choice(range(len(meths)),p=p)
                meth=meths[mi]
                escore=meth()
                if escore==None:
                    continue
                e,score=escore

                if score < bestscore:
                    beste,bestscore,tt=e,score,tracer.tracetree

                    print('%d %8.2f : %s' % (mi,bestscore, beste))
            if tl.over().total_seconds()<.1:
                print('   %8.2f %8.2f %s' % (bestscore, factor, ''.join('%8.3f' % x for x in  p)))
            sys.stdout.flush()
    except KeyboardInterrupt:
        print('\nctrl-c pressed')


def demo_tracetree_analysis():
    import random
    g=ArithmeticGrammar()
    sampler=GrammaSampler(g)
    tracer=Tracer()
    sampler.add_sideeffects(tracer)

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
    class GStub(GrammaGrammar):
        def __init__(self,**kw):
            GrammaGrammar.__init__(self,'start := "";',**kw)

    def eval_grammar(Grammar,expect):
        # if a state variable is used by a gfunc, it must be reset
        s=None
        try:
            g=Grammar()
        except Exception as e:
            s=str(e)
        if (s==None and expect==None) or (s!=None and expect!=None and expect in s):
            print('Checked: %s' % Grammar.__name__)
        else:
            print('check failed for %s. expect:\n  %s\ngot:\n  %s' % (Grammar.__name__, expect, s))

    class AnalyzeMeGrammar1(GStub):
        @gfunc
        def f(x):
            yield str(x.state.m)
    eval_grammar(AnalyzeMeGrammar1,'x.state.m used without being initialized in any reset_state')

    class AnalyzeMeGrammar1fix(GStub):
        def reset_state(self,state):
            state.m=5
        @gfunc
        def f(x):
            yield str(x.state.m)
    eval_grammar(AnalyzeMeGrammar1fix,None)


    # global are forbidden..
    class AnalyzeMeGrammar2(GStub):
        @gfunc
        def f(x):
            yield str(g_global)
    eval_grammar(AnalyzeMeGrammar2, 'forbidden access to variable "g_global"')

    # unless explicitly allowed...
    class AnalyzeMeGrammar2fix(GStub):
        def __init__(self):
            GStub.__init__(self,allowed_global_ids=['g_global'])
        @gfunc
        def f(x):
            yield str(g_global)
    eval_grammar(AnalyzeMeGrammar2fix,None)


    # gfuncs don't return their value, they yield it.
    class AnalyzeMeGrammar3(GStub):
        @gfunc
        def f(x):
            return 'my value'
    eval_grammar(AnalyzeMeGrammar3, "gfunc f of class AnalyzeMeGrammar3: doesn't yield a value")

    class AnalyzeMeGrammar3fix(GStub):
        @gfunc
        def f(x):
            yield 'my value'
    eval_grammar(AnalyzeMeGrammar3fix,None)

    # gfuncs don't return their value, they yield it.
    class AnalyzeMeGrammar3b(GStub):
        @gfunc
        def f(x,ge):
            s=yield ge
            s+=yield (ge)
            s+=yield 'not an argument'
            yield  s
    eval_grammar(AnalyzeMeGrammar3b, "gfuncs can only sample from their arguments")
    


    class AnalyzeMeGrammar4(GStub):
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
        def f(x,ge):
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


            s=yield ge
            yield s

    g=AnalyzeMeGrammar4()
    def got_expected(Grammar,s,got,expect):
        if got==expect:
            print('Checked: %s %s' % (Grammar.__name__,s))
        else:
            print('check failed for %s %s. expect:\n  %s\ngot:\n  %s' % (Grammar.__name__, s, expect, got))

    got_expected(AnalyzeMeGrammar4, 'uses', ','.join(sorted(g.funcdefs['f'].meta.statevar_uses)),'mod,obj1,obj3,subscript_mod,subscript_mod2,subscript_use,used')
    got_expected(AnalyzeMeGrammar4, 'defs', ','.join(sorted(g.funcdefs['f'].meta.statevar_defs)),'assigned,mod,obj1,obj2,obj3,subscript_def,subscript_mod,subscript_mod2')

    class AnalyzeMeGrammar5(GStub):
        @gfunc
        def f(x):
            yield ''.join([(yield 'e%d') for e in range(3)])
    eval_grammar(AnalyzeMeGrammar5, 'yield in a generator expression or list comprehension')

    class AnalyzeMeGrammar5a(GStub):
        @gfunc
        def f(x,**kw):
            yield ''
    eval_grammar(AnalyzeMeGrammar5a, "gfuncs mustn't take keyword arguments")

    class AnalyzeMeGrammar5b(GStub):
        @gfunc
        def f(x,y=1):
            yield ''
    eval_grammar(AnalyzeMeGrammar5b, "gfuncs mustn't use default argument values")

    class AnalyzeMeGrammar6(GrammaGrammar):
        def __init__(self):
            GrammaGrammar.__init__(self, '''
                start := `abc`?"a":"b";
            ''')
    eval_grammar(AnalyzeMeGrammar6, 'abc used without being initialized in any reset_state')

    class AnalyzeMeGrammar6fix(GrammaGrammar):
        def __init__(self):
            GrammaGrammar.__init__(self, '''
                start := `abc`?"a":"b";
            ''')
        def reset_state(self, state):
            state.abc=False
    eval_grammar(AnalyzeMeGrammar6fix, None)


    class AnalyzeMeGrammar7(GrammaGrammar):
        def __init__(self):
            GrammaGrammar.__init__(self, '''
                start := `abc` "a" | "b";
            ''')
    eval_grammar(AnalyzeMeGrammar7, 'abc used without being initialized in any reset_state')

    class AnalyzeMeGrammar8(GrammaGrammar):
        def __init__(self):
            GrammaGrammar.__init__(self, '''
                start := "a"{`abc`};
            ''')
    eval_grammar(AnalyzeMeGrammar8, 'abc used without being initialized in any reset_state')


def demo_meta():
    g=DemoGrammar()
    ge=g.parse('''
             geta()
           . "=0."
           . def("a",`9`)
           . geta()
           . "=9."
           . load("a","a0")
           . geta()
           . "=13."
           . incra()
           . geta()
           . "=14."
           . save("a","a1")
    ''')

    sampler=GrammaSampler(g)
    sampler.update_statecache(a0=13)
    s=sampler.sample(ge)
    print(s)
    print(sampler.get_statecache())
    def r(ge,indent=0):
        print((' '*indent) + '%s[%s]' % (ge,ge.get_meta()))

        if isinstance(ge,GInternal):
            for c in ge.children:
                r(c,indent+1)
    print('node[meta]')
    r(ge)

def demo_timing():
    import time

    g=ArithmeticGrammar()
    sampler=GrammaSampler(g)
    sampler.random.seed(0)
    n=2500
    #n=10000
    t0=perf_counter()
    for i in xrange(n):
        s=sampler.sample()
    t1=perf_counter()
    print('avg=%f' % ((t1-t0)/n))

def demo_profile():
    import cProfile
    cProfile.run('demo_timing()')



def demo_params():
    #  anonymous grammar with some gcode using a param
    g=GrammaGrammar('''
        start:="a"{`x`};
    ''',param_ids=['x'])
    sampler=GrammaSampler(g)
    sampler.update_params(x=7)
    assert(sampler.sample()=='a'*7)
    sampler.update_params(x=3)
    assert(sampler.sample()=='a'*3)

    # named class with a gfunc using a param
    class ParamsGrammar(GrammaGrammar):
        G='''
            start:=f();
        '''
        def __init__(self):
            GrammaGrammar.__init__(self, self.G, param_ids=['x'])
        @gfunc
        def f(x):
            yield str(x.params.x)

    g=ParamsGrammar()
    sampler=GrammaSampler(g)
    sampler.update_params(x=7)
    assert(sampler.sample()=='7')
    sampler.update_params(x=3)
    assert(sampler.sample()=='3')


if __name__=='__main__':
    demo_parser()
    demo_sampling()
    demo_recursion_limits()
    demo_tracetree()
    demo_transform()
    demo_random_states()
    demo_resample()
    demo_resample_funcy()
    demo_tracetree_analysis()
    demo_grammar_analysis()
    demo_meta()
    demo_params()

    #demo_fit()

    #demo_timing()
    #demo_profile()


# vim: ts=4 sw=4

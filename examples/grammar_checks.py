#!/usr/bin/env python
'''

    demonstrate some of the checking that gramma does to ensure that gfuncs are
    well-formed.

'''
from __future__ import absolute_import, division, print_function
from builtins import super

from gramma2 import *

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



# state variables must be initialized
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


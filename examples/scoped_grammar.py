#!/usr/bin/env python3
"""

    demonstrate scoped variable declaraions

"""
from __future__ import absolute_import, division, print_function

from gramma import *


class Defaulted(object):
    def __init__(self, default_value):
        self._default_ = default_value

    def __getattr__(self, name):
        return self._default_


class Scoping(SideEffect):
    def reset_state(self, state):
        state.d = 0
        state.scope_stack = []
        state.scope = None

    def push(self, x, ge):
        if isinstance(ge, GRule) and ge.rname == 'block':
            x.state.d += 1

            stk = x.state.scope_stack
            # initalize new scope with vars and has_vars
            sc = Defaulted(False)
            sc.vars = []
            # if any parent has vars, so do we
            if len(stk) > 0 and stk[-1].has_vars:
                sc.has_vars = True

            # update state
            x.state.scope = sc
            stk.append(sc)
            return True
        return False

    def pop(self, x, w, s):
        if w:
            x.state.scope_stack.pop()
            if len(x.state.scope_stack) > 0:
                x.state.scope = x.state.scope_stack[-1]
            x.state.d -= 1


class ScopedVariableGrammar(GrammaGrammar):
    G = r'''
        start := "-----\n".block;

        # indent according to block rule depth
        spc := "  ";
        ind := spc{`d-1`};

        expr :=   "def ".new(var).";"
                | `5*scope.has_vars` "use ".var().";"
                | `d<5` block  # limit recursion depth
                | .1 "nop;"
                ;

        block := "{\n".                          # (indent and..) opening brace
                 (ind.spc.expr."\n"){1,5}.
                 ind."}";                            # closing brace.

        # variable names are 1 to 5 letters, geomterically distributed w/ a mean of 3
        var := ['a'..'z']{1,5,geom(3)} ;
    '''

    def __init__(x):
        GrammaGrammar.__init__(x, type(x).G, sideeffects=[Scoping()])

    @gfunc
    def new(x, ge):
        # generate a name with the argument
        n = yield ge
        # and add it to the current scope's vars list
        x.state.scope.vars.append(n)
        # and update has_vars
        x.state.scope.has_vars = True
        yield n

    @gfunc
    def var(x):
        # collect all variable names from this and enclosing scopes
        lvars = []
        for sc in x.state.scope_stack:
            lvars.extend(sc.vars)
        # and pick one
        yield x.random.choice(lvars)


sampler = GrammaSampler(ScopedVariableGrammar())
for i in range(3):
    print(sampler.sample())

# vim: ts=4 sw=4

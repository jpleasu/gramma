#!/usr/bin/env python3
"""

demonstration of a grammar with scoped conditions and guarded rules

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
        state.scope_stack = []
        state.scope = None

    def push(self, x, ge):
        if isinstance(ge, GRule) and ge.rname == 'block':
            x.state.scope_stack.append(Defaulted(False))
            x.state.scope = x.state.scope_stack[-1]
            return True
        return False

    def pop(self, x, w, s):
        if w:
            x.state.scope_stack.pop()
            if len(x.state.scope_stack) > 0:
                x.state.scope = x.state.scope_stack[-1]


class GuardedRules(GrammaGrammar):
    '''
        scopes are bounded by '{' and '}'
        a scope can be "initalized" at most once with "init;"
        only after initalization, can a scope contain "cmd;"
        a "nop;" can be run without initialization

        e.g. 
            {init;cmd;}   <- ok
            {init;nop;}   <- ok
            {nop;init;}   <- ok
            {init;init;}  <- not ok
            {cmd;init;}   <- not ok
    '''

    G = r'''
        start := block;
        block := '{'. expr{1,4} . '}';
        expr :=   `not scope.initialized` 'init;' . set('initialized')
                | 'nop;'
                | block
                | `scope.initialized` 'cmd;';
     '''

    def __init__(self):
        GrammaGrammar.__init__(self, self.G, sideeffects=[Scoping()])

    @gfunc
    def set(x, n):
        setattr(x.state.scope, n.as_str(), True)
        yield ''


if __name__ == '__main__':
    g = GuardedRules()
    s = GrammaSampler(g)
    for i in range(10):
        print(s.sample())

# vim: ts=4 sw=4

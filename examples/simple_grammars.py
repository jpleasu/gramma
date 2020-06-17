#!/usr/bin/env python3
"""

simple grammars -- used in other examples

"""
import traceback

import gramma
from gramma import *

# "global" means the globals in the gramma module, e.g. vars(gramma)
# unless we change gcode_globals to point here, as follows:

gramma.gcode_globals = globals()


# now these will be visible in gcode
def g_func():
    for line in traceback.format_stack():
        print(line)
    return 8


g_allowed = 'g_allowed'


class ArithmeticGrammar(GrammaGrammar):
    G = r'''
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
        GrammaGrammar.__init__(x, type(x).G, sideeffects=[DepthTracker])


class VarietyGrammar(GrammaGrammar):
    '''
         a grammar that uses a variety of gramma's features
    '''

    G = r'''
        start := recurs;

        yyy:='a'{,`g_func()`};

        xxx:='a'{1,`g_func()`};

        recurs := 10 ".".recurs | vars . " ". words . " " . ['1'..'9'] . digit{1,15,geom(5)};

        digit := ['0' .. '9'];

        weirddigits := ['0','2'..'5','7','9'];

        words := (`1000*(depth>20)` "*" | " ").
                 ( .75 "dog" | .25 "cat" ).
                 (" f=".f()." a=".(`a`?"1":"0")." ff=".ff()){1,4};

        vars := choose x~('a'|'b') in x.x.x;
    '''

    def __init__(self):
        GrammaGrammar.__init__(self, self.G, sideeffects=[DepthTracker], allowed_global_ids='g_allowed g_func'.split(),
                               param_ids='maxrep'.split())

    def reset_state(self, state):
        super().reset_state(state)
        state.a = 0
        state.x = type('_', (), {})
        state.x.y = 7
        state.extrastate = {}

    @gfunc
    def f(x):
        yield x.random.choice(['woof', 'meow'])

    @gfunc
    def ff(x):
        x.state.a ^= 1
        yield ['bleep', 'bloop'][x.state.a]

    @gfunc()
    def use_global(x):
        yield 'use_global_return ' + g_allowed

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
        x.state.extrastate = 7
        yield ''


if __name__ == '__main__':
    print(GrammaSampler(VarietyGrammar(), maxrep=100).sample())

# vim: ts=4 sw=4

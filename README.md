# Table of contents

* [Description](#description)
* [Overview](#overview)
* [Install](#install)
* [samplers](#samplers)
    * [Python](#python)
* [C++](#c)
* [GLF syntax](#glf-syntax)
    * [literals - same syntax as Python strings](#literals---same-syntax-as-python-strings)
    * [ranges - (`[` .. `]`)  - character ranges](#ranges---------character-ranges)
    * [ternary operator (`?:`) - choice based on computed boolean](#ternary-operator----choice-based-on-computed-boolean)
    * [weighted alternation (`|`) - weighted random choice from alternatives](#weighted-alternation----weighted-random-choice-from-alternatives)
    * [concatenation (`.`) - definite concatenation](#concatenation----definite-concatenation)
    * [repetition (`{`...`}`) - random repeats](#repetition----random-repeats)
    * [variables (`choose ` .. `~` .. `in` ...) - reuse of samples](#variables-choose-----in----reuse-of-samples)
    * [function call (gfuncs) - defined outside of the GLF](#function-call-gfuncs---defined-outside-of-the-glf)
* [creating grammars](#creating-grammars)
    * [from Antlr4](#from-antlr4)
    * [preventing explosion from recursion](#preventing-explosion-from-recursion)
* [other topics](#other-topics)
    * [`TraceTree`](#tracetree)
    * [Resampling](#resampling)
* [Rope Aplenty](#rope-aplenty)
    * [`TraceNode.child_containing`](#tracenodechild_containing)
    * [`TraceNode.resample`](#tracenoderesample)

# Description

Gramma is a probabilistic programming language for grammar based fuzzing.

# Overview
Expressions in Gramma are probabilistic programs with string value.  They
are written in GLF, an extensible syntax for formal language description
that resembles extend Backus-Naur form (EBNF).  Gramma is like a parser generator, 
but instead of generating a parser from a grammar, Gramma generates a (parameterized) fuzzer
from a grammar.

A typical application of Gramma in fuzzing would be as follows:

1. Create a GLF grammar based on the input grammar for the application under test.
2. Feed the instrumented application samples and compute a measure of interest
   for each.
3. Tweak numerical parameters of the grammar and/or use previous samples as
   templates to update the grammar.
4. Repeat.


# Install

Gramma is pure Python 3 with some dependencies.

```
pip3 install lark-parser six future numpy
```

# samplers
GLF expressions aren't evaluated, they're _sampled_.  Instead of an execution engine, interpreter, or compiler,
we need a _sampler_ to get a string from a GLF expression.

## Python
The Python sampler is an interpreter, built for analysis of the language itself.
```python
from gramma import *

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

f __name__ == '__main__':
    print(GrammaSampler(ArithmeticGrammar()).sample())
```

# C++
C++ samplers are generated for speed.
see [cppgen](tools/cppgen/README.md)


# GLF syntax

GLF, the gramma language format, is structurally the same as BNF with different syntax for the operators and 
some extra features:
- GLF permits "gcode" and "gfunc" terms which hold the place for bits of the sampler implemented elsewhere, 
e.g. in Python or C++.
- GLF expressions are untyped, it's up to the sampler to interpret.

## literals - same syntax as Python strings
Literals are parsed as Python strings:
```
'this is a literal'
"""this is a 
    multiline literal"""
```

## ranges - (`[` .. `]`)  - character ranges
```
['a'..'z']
```
samples a character uniformly from `a` to `z`, inclusive.

Multiple subranges and single characters can be included in the set to be
sampled uniformly:
```
['a'..'z', '0'..'9', '_']
```

## ternary operator (`?:`) - choice based on computed boolean
```
`depth<5` ? x : y
```
The "code" term, `depth<5`, is a Python expression testing the state variables `depth`. 
If the computed result is `True`, the result is `x`, else `y`.
## weighted alternation (`|`) - weighted random choice from alternatives
```
2 x | 3.2 y | z
```
selects one of `x`, `y`, or `z` with probability `2/5.3`, `2.3/5.3`, and `1/5.3` respectively.  The weight 
on `z` is implicitly `1`.

Omitting all weights corresponds to flat random choice, e.g.
```
x | y | z
```
selects one of x,y, or z with equal likelihood.

Weights can also be "dynamic", code written in backticks. For example:
```
recurs := `depth<5` recurs | "token";
```  
this sets the recurse branch weight to `1` if `depth <5`, and `0` otherwise (because `int(True)==1` and `int(False)==0`).
              
## concatenation (`.`) - definite concatenation
```
x . y
```
concatenates `x` and `y`.


## repetition (`{`...`}`) - random repeats
```
x{3}
```
generates `x` exactly `3` times.
  
If `x` is non constant in the above expression, we can get `3` different things, e.g.
```
("a"|"b"|"c"){3}
```
can generate `aaa`, `aab`, `aac`, ..., `ccc`.

```
x{1,3}
```
generates a number `n` uniformly in the closed interval `[1,3]` then generates `x` exactly `n` times.

```
x{geom(3)}
```
samples a number `n` from a geometric distribution with mean `3`, then generates `x` exactly `n` times.

```
x{3,,geom(3)}
```
same as above, but reject `n` less than `3`.

```
x{1,5,geom(3)}
```
same as above, but reject `n` outside of the interval `[1,5]`.

Bounds can be gcode to compute at runtime, e.g.
```
x{1,`maxrep`}
```
generates `x` between `1` and `maxrep` times, where `maxrep` is a state variable or parameter.

## variables (`choose ` .. `~` .. `in` ...) - reuse of samples
```
choose v ~ ("a"|"b")  in  v.v.v.v
```
samples `v` once and uses the result `4` times. The result is either `aaaa` or `bbbb`.

Multiple variables can be chosen in one statement:
```
choose v1~x, v2~y  in (v1.v2)
```
note the scoping though - if `y` contains a reference to `v1`, it will be resolved in the containing scope,
it won't use the sample of `x`.

The choose keyword can be omitted, e.g.
```
v1~x,v2~y in v1.v2
```

## function call (gfuncs) - defined outside of the GLF
```
f(x)
```
- in Python, inherit from the `GrammaGrammar` class and add decorated functions, see below.
- functions can be stateful, meaning they rely on information stored in the `SamplerInterface` object.
- evaluation is left to right.
     
# creating grammars

## from Antlr4 
see [g42glf](tools/g42glf/README.md)

## preventing explosion from recursion
To identify rules that are being visited excesesively, count (and emit) rule
hits while sampling with sideeffect - see `StackWatcher` in the [smtlibv2
example](examples/smtlib2/smtlibv2.py).

To avoid rules, you can weight alternations to avoid them anywhere in the loop
of rule name references. You can also compute depth (with sideffect) and use a
dynamic alternation to avoid looping references.


# other topics

## `TraceTree`
Sampling in Gramma is a form of expression tree evaluation where each node
can use a random number generator.  E.g. to sample from

```
"a" | "b"{1,5};
```

The head of this expression, alternation, randomly chooses between its
children, generating either "a" or a random sample of "b"{1,5} with equal
odds.

If "a" is chosen, the sample is complete.  If "b"{1,5} is chosen, its head,
the repetition node, draws a random count between 1 and 5, then samples its
child, "b", that number of times.  The results are concatenated, returned
to the parent alternation node, and the sample is complete.

Each possibility results in a different tree. Pictorially,

```
    alt       or         alt
     |                    |
    "a"                  rep
                        /...\
                       |     |
                      "b"   "b"
```

The Tracer sideeffect computes this, so called, "trace tree", storing
random number generator and other state when entering and exiting each
node.

Note: When sampling from a rule, the trace tree resembles the "recursion
tree" of the recursion tree method for evaluating recursive programs. For
example, a sample from

    r := "a" | "b" . r;

could produce the trace tree

```
    alt
     |
    cat
    / \
   |   |
  "b" rule(r)
       |
      alt
       |
      "a"
```

where the first alternation selected "b" . r, a concatenation whose
righthand child samples the rule r recursively.


## Resampling

To produce strings similar to previous samples, we can hold fixed (or
definitize) part of the corresponding trace tree.  By choosing to replay
the context of execution for some but not all nodes of the trace tree, we
can effectively create a template from a sample.

The TraceNode object computed by the Tracer sideffect provides an interface
for performing the operations presented below.

- "resampling" starts with a tracetree and a node. For example, given the `GExpr`
```
a(b(),c(),d())
```
To record the random context around `c` we compute:
```
save_rand('r0').a(b(),c().save_rand('r1'),d())
```
and store `r0` and `r1`.  The random number generator on entering `c` is then reseeded on entry, and resumed with 
`r1` after exiting `c`.

```
load_rand('r0').a(b(), reseed_rand().c().load_rand('r1'), d())
```

we could also choose the reseed explicitly via a load, e.g. if we'd saved a random state `r2` we could use:
```
load_rand('r0').a(b(), load_rand(r2).c().load_rand('r1'), d())
```
to handle recursion, we must "unroll" rules until the point where the resampled node occurs.  e.g. the 
following generates arrows, `---->` with length (minus 1) distributed geometrically.
```
r:= "-".r | ">";
```

To resample the `r` node that's three deep in the trace tree of `----->`, we partially unroll the expression `r`:

```
"-".("-".("-".r | ">") | ">") | ">";
```
        
and instrument:
```
save_rand('r0').("-".("-".("-"
    .r.save_rand('r1') | ">") | ">") | ">");
```
then replay with a reseed
```
load_rand('r0').("-".("-".("-"
    .reseed_rand().r.load_rand('r1') | ">") | ">") | ">");
```

# Rope Aplenty
GFuncs are (nearly) arbitraty code, so the analysis done by Gramma is
necessarily limited.  To understand when Gramma might "give up", examples
are given here.

"Well behaved" GFuncs are generally subsumed by other constructs in Gramma,
like ternary operators or dynamic alternations.  Heuristics, therefore,
which would apply to "well behaved" gfuncs are avoided, assuming that
grammars using gfuncs really need them.

## `TraceNode.child_containing`
.. treats gfuncs as atomic. a gfunc can modify the strings that it samples,
so Gramma can't tell what parts of the string sampled from a gfunc are from
what child.

## `TraceNode.resample`
When a child node of a gfunc is resampled, Gramma tries to match previously
sampled arguments with gexpr children of the original call, so that a new
call can be constructed with appropriately sampled/definitized arguments.

e.g. suppose we have
```
select(X,Y,Z)
```

where our grammar defines
```python
    @gfunc
    def select(x, a, b, c):
        if (yield a)=='y':
            yield (yield b)
        else:
            yield (yield b)
```

The tracetree records the sampled `X` and _depending on its value_ either the sampled `Y` or the sampled `Z`, but 
not both.

If we resample (a child of) `X`, and suppose the original sample had chosen `Y`. Gramma will use the definitized sample 
of `Y` in the 2nd argument and the original expression for `Z` in the 3rd.

If we resample (a child of) `Y`, `Y` must have been sampled.. so `Z` was not.. we will use the previous sample for 
`X`, which will again select `Y`.. what we use in the 3rd argument doesn't matter in this case, but Gramma will use
the original Z.

If we resample `X` and `Y`, then it's possible that `Z` is sampled, since the 1st arg might select differently.



If an argument is sampled more than once by a gfunc, that's a different story. suppose we have
```    
bigger("a"{0,5})
```    
where our grammar defines
```python
    @gfunc
    def bigger(x, a):
        a1=(yield a)
        a2=(yield a)
        yield (a1 if len(a1)>len(a2) else a2)
```    
Suppose we resample the longer `"a"{0,5}` sample. Without replaying the
previous sample, there's no way to reproduce the function's behavior.  In
this case, we therefore resample the entire argument.


# Table of contents

* [Description](#description)
* [Overview](#overview)
* [Setup](#setup)
* [samplers](#samplers)
    * [Python](#python)
    * [C++ sampler generator](#c-sampler-generator)
* [GLF syntax](#glf-syntax)
    * [literals - same syntax as Python strings](#literals---same-syntax-as-python-strings)
    * [ranges - (`[` .. `]`)  - character ranges](#ranges---------character-ranges)
    * [ternary operator (`?:`) - choice based on computed boolean](#ternary-operator----choice-based-on-computed-boolean)
    * [weighted alternation (`|`) - weighted random choice from alternatives](#weighted-alternation----weighted-random-choice-from-alternatives)
    * [denotation (`/`) - denotation](#denotation----denotation)
    * [concatenation (`.`) - definite concatenation](#concatenation----definite-concatenation)
    * [repetition (`{`...`}`) - random repeats](#repetition----random-repeats)
    * [variables (`choose ` .. `~` .. `in` ...) - reuse of samples](#variables-choose-----in----reuse-of-samples)
    * [rules](#rules)
        * [parameterized rules](#parameterized-rules)
    * [function call (gfuncs) - defined outside of the GLF](#function-call-gfuncs---defined-outside-of-the-glf)
* [creating grammars](#creating-grammars)
    * [from Antlr4](#from-antlr4)
    * [preventing explosion from recursion](#preventing-explosion-from-recursion)
* [other topics](#other-topics)
    * [`TraceTree`](#tracetree)
    * [unrolling](#unrolling)

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


# Setup

Gramma is pure Python 3, depending on `lark-parser` and `numpy`.

```bash
git clone git@github.com:jpleasu/gramma.git
cd gramma

# to install
pip3 install .
```

For developement
```bash
# to run using repo contents (instead of copying to site-packages)
pip3 install -e .

# to nstall extra testing dependencies
pip3 install -e .[tests]

# to run tests
tox

# coverage summary is at ./htmlcov/index.html
```


# samplers
Gramma is like an ordinary programming language, except it isn't evaluated, it's _sampled_.  Instead of an execution 
engine, interpreter, or compiler, we use a _sampler_ to get a string from a GLF expression.

## Python
The Python interpretering sampler provides a fast way to prototype a grammar with a straightforward interpreter.  If 
the interpreter is exceeding Python's stack depth, a coroutine based interpretering sampler is also provided.  It's a 
bit harder to follow, but it can be worth the effort for debugging hairy grammars.

```python
from gramma.samplers import GrammaInterpreter, gfunc, gdfunc, Sample
class Arithmetic(GrammaInterpreter):
    GLF = '''
        start := expr;
        expr := add;
        add := mul . ('+'.mul){,3};
        mul := atom . ('*'.atom){,3};
        atom :=   'x'
                | randint()
                | `expr_rec` '(' . expr . ')';
    '''

    def __init__(self, glf=GLF):
        super().__init__(glf)
        self.expr_rec = .1

    @gfunc
    def randint(self):
        return self.create_sample(str(self.random.integers(0, 100000)))

if __name__ == '__main__':
    sampler=Arithmetic()
    print(sampler.sample_start())
```

## C++ sampler generator
The C++ sampler generator produces C++17 source that depends on `include/gramma/gramma.hpp`.

`glf2cpp` can generate a template sampler class and `main` to produce samples with it, but non-trivial grammars,
in particular those with denotations, will require more work.

The generated samplers use template parameters `ImplT`, `SampleT`, and `DenotationT` in a CRTP pattern, where 
the implementation, `ImplT` is expected to provide `cat` and `denote` methods.

```bash
glf2cpp --help
```

TODO: example

### function argument evaluation order
Unfortunately, argument evaluation order cannot be prescribed in C++. In particular, we can't assume that `g1` is 
executed first in the following expression:
```C++
f(g1(), g2(), g3());
```
We can force the order by injecting sequence points:
```C++
auto &&a1=g1();
auto &&a2=g2();
auto &&a3=g3();
f(std::move(a1), std::move(a2), std::move(a3));
```
but there is a small performance hit -- the compiler can no longer perform copy/move elision.

The `glf2cpp` option `--enforce_ltr` will generate this extra code.

Even still, the converting constructor of `sample_t` might run into this as well. Calls to gfuncs are generated 
with callable arguments:
```C++
f(g1,g2,g3)
```
This allows gfunc implementations to avoid ever sampling subexpressions.  It also allows the gfunc to prescribe 
the evaluation order.

For simpler gfunc implementations, `sample_t` has a coverting constructor, from callable to sample.  E.g. the following
prototype for a gfunc is fine:
```C++
sample_t f(sample_t a1, sample_t a2, sample_t a3);
```
The problem is that the compiler will invoke the callable converting-constructor of `sample_t` for each argument, in
the compiler's arbitrary argument evaluation order.

So, **if the order of argument sampling matters** then use `--enforce_ltr` and define your gfuncs with 
`sample_factor_type` argument types:
```C++
sample_t f(sample_factory_type a1f, sample_factory_type a2f, sample_factory_type a3f) {
    auto &&a1=a1f();
    auto &&a2=a2f();
    auto &&a3=a3f();

    return a1+a2+a3;
}
```



# GLF syntax

GLF, the gramma language format, is structurally the same as BNF with different syntax for the operators and 
some extra features:
- GLF permits "gcode" and "gfunc" terms which hold the place for bits of the sampler implemented elsewhere, 
e.g. in Python or C++.
- GLF expressions are untyped, it's up to the sampler.

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
              

## denotation (`/`) - denotation
```
    x / y
```
Denotes the value `x` with `y`.

The following pseudo code demonstrates how a sampler would interpret the expression above:
```
    sampler.sample('x/y') -> sampler.sample('x').denote('y')
```

We can interpret the syntax, `-/-` as defining the mapping relation for denotational semantics of the generated 
language.  Semantics can influence sampling by accessing denotations in gfuncs.

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

## rules
```
r := '->' . ('stop' | r);
```
Rules provide recursion in gramma.  Care must be taken to avoid runaway recursion, e.g. by weighting the recursing 
option with a small weight:
```
r := ('->'|'=>') . ('stop' | .001 r);
```


### parameterized rules
```
q(a,b) := a . (b | q(a,b));

not_r := q('->'|'=>','stop');
```
Rules can take parameters which are bound on call. In particular, `not_r` is not the same thing as `r` in the previous
grammar, because `'->'|'=>'` is sampled and bound *once* when `not_r` is invoked.


## function call (gfuncs) - defined outside of the GLF
```
f(x)
```
- in Python, inherit from the `GrammaGrammar` class and add decorated functions, see below.
- functions can be stateful, meaning they rely on information stored in the `SamplerInterface` object.
- evaluation is left to right.
     
# creating grammars

## from Antlr4 
In some cases, an [ANTLR4 grammar](https://github.com/antlr/grammars-v4) can be a good starting place, so the tool
`g4toglf` is provided.

## preventing explosion from recursion
TODO: rewrite this..

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

## unrolling
TODO

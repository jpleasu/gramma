# Table of contents

* [Description](#description)
* [Overview](#overview)
* [Install](#install)
* [GLF Syntax](#glf-syntax)
* [other topics](#other-topics)
    * [TraceTree](#tracetree)
    * [Resampling](#resampling)
* [Rope Aplenty](#rope-aplenty)
    * [TraceNode.child_containing](#tracenodechild_containing)
    * [TraceNode.resample](#tracenoderesample)


# Description

Gramma is a probabilistic programming language for grammar based fuzzing.


# Overview
Expressions in Gramma are probabilistc programs with string value.  They
are written in GLF, an extensible syntax for formal language description
that resembles Backus-Naur form (BNF).

GLF is extended with custom functions implemented in extensions of the base
`GrammaGrammar` class.

A typical application of Gramma in fuzzing would be as follows:

1. Create a grammar based on the input grammar for the application under test.
2. Feed the instrumented application samples and compute a measure of interest
   for each.
3. Tweak numerical parameters of the grammar and/or use previous samples as
   templates to update the grammar.
4. Repeat.


# Install

Gramma is pure python 3 with some dependencies.

```
pip3 install lark-parser six future numpy
```

# GLF Syntax
```
    literals - same syntax as Python strings
        'this is a string'
        """this is a (possibly 
            multiline) string"""

    ternary operator (?:) - choice based on computed boolean
        `depth<5` ? x : y
        - the code term, `depth<5`, is a python expression testing the state
          variables 'depth'.  If the computed result is True, the result is
          x, else y.

    weighted alternation (|) - weighted random choice from alternatives
        2 x | 3.2 y | z
        - selects on of x, y, or z with probability 2/5.3, 2.3/5.3, and 1/5.3
          respectively.  The weight on z is implicitly 1.
        - omitting all weights corresponds to flat random choice, e.g.
            x | y | z
          selects one of x,y, or z with equal likelihood.
        - weights can also be code in backticks. For example:
            recurs := `depth<5` recurs | "token";
            - this sets the recurse branch weight to 1 (int(True)) if depth <5,
              and 0 otherwise.

    concatenation (.) - definite concatenation
        x . y

    repetition ({}) - random repeats
        x{3}
            - generate x exactly 3 times
        x{1,3}
            - generate a number n uniformly in [1,3] then generate x n times
        x{geom(3)}
            - sample a number n from a geometric distribution with mean 3, then
              generate x n times
        x{3,,geom(3)}
            - same as above, but reject n less than 3
        x{1,5,geom(3)}
            - same as above, but reject n outside of the interval [1,5]

        - furthermore, bounds can be replaced with gcode to compute at runtime,
          e.g.

          x{1,`maxrep`}
            - generate x between 1 and eval(`maxrep`) times

     function call (gfuncs) - as defined in a GrammaGrammar subclass
        f(x)

        - by inheriting the GrammaGrammar class and adding decorated functions,
          GLF syntax can be extended.  See below.
        - functions can be stateful, meaning they rely on information stored in
          the SamplerInterface object.
        - evaluation is left to right.
        - functions aren't allowed to "look up" the execution stack, only back.
```
# other topics

## TraceTree
Sampling in Gramma is a form of expression tree evaluation where each node
can use a random number generator.  E.g. to sample from

    "a" | "b"{1,5};

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

where the first alternation selected "b" . r, a concatenation whose
righthand child samples the rule r recursively.


##Resampling

To produce strings similar to previous samples, we can hold fixed (or
definitize) part of the corresponding trace tree.  By choosing to replay
the context of execution for some but not all nodes of the trace tree, we
can effectively create a template from a sample.

The TraceNode object computed by the Tracer sideffect provides an interface
for performing the operations presented below.

- "resampling" starts with a tracetree and a node. For example, given the
  GExpr
```

        a(b(),c(),d())

    To record the random context around "c" we compute:

        save_rand('r0').a(b(),c().save_rand('r1'),d())

    and store r0 and r1.  The random number generator on entering "c" is
    then reseeded on entry, and resumed with r1 after exiting "c".

        load_rand('r0').a(b(), reseed_rand().c().load_rand('r1'), d())

    we could also choose the reseed explicitly via a load, e.g. if
    we'd saved a random state "r2" we could use:

        load_rand('r0').a(b(), load_rand(r2).c().load_rand('r1'), d())
```

- to handle recursion, we must "unroll" rules until the point where the
  resampled node occurs.  e.g. the following generates arrows, "---->"
  with length (minus 1) distributed geometrically.
```
        r:= "-".r | ">";

    To resample the "r" node that's three deep in the trace tree of
    "----->", we partially unroll the expression "r":

        "-".("-".("-".r | ">") | ">") | ">";
        
    and instrument:

        save_rand('r0').("-".("-".("-"
            .r.save_rand('r1') | ">") | ">") | ">");
        
    then replay with a reseed

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

## TraceNode.child_containing
.. treats gfuncs as atomic. a gfunc can modify the strings that it samples,
so Gramma can't tell what parts of the string sampled from a gfunc are from
what child.

## TraceNode.resample
When a child node of a gfunc is resampled, Gramma tries to match previously
sampled arguments with gexpr children of the original call, so that a new
call can be constructed with appropriately sampled/definitized arguments.
```
    e.g. suppose we have

        choose(X,Y,Z)

    where

        @gfunc
        def choose(x, a, b, c):
            if (yield a)=='y':
                yield (yield b)
            else:
                yield (yield b)

    The tracetree records the sampled X and _depending on it's value_
    either the sampled Y or the sampled Z, but not both.

    If we resample (a child of) X, and suppose the original sample had
    chosen Y. Gramma will use the definitized sample of Y in the 2nd
    argument and the original expression for Z in the 3rd.

    If we resample (a child of) Y, Y must have been sampled.. so Z was
    not.. we will use the previous sample for X, which will again choose
    Y.. what we use in the 3rd argument doesn't matter in this case, but
    Gramma will use the original Z.

    If we resample X and Y, then it's possible that Z is sampled, since the
    1st arg might choose differently.
```
If an argument is sampled more than once by a gfunc, that's a different story.
```    
    suppose we have

        bigger("a"{0,5})

    where

        @gfunc
        def bigger(x, a):
            a1=(yield a)
            a2=(yield a)
            yield (a1 if len(a1)>len(a2) else a2)

    Suppose we resample the longer "a"{0,5} sample. Without replaying the
    previous sample, there's no way reproduce the function's behavior.  In
    this case, we therefore resample the entire argument.
```

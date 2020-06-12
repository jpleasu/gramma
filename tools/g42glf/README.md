# setup

1. read [setup.sh](setup.sh)
2. run `setup.sh`

# usage
e.g. for json
``` bash
  ./g4toglf.py grammars-v4/json/JSON.g4 2> JSON.glf
```

ANTLR grammars are for parsing and not generation, so certain constructs need
tweaking.

- in ANTLR, `X*` will recognize any sequence of `X`'s... to generate, we need
  a distribution on the number of repetitions.  g4toglf uses a parameter
  `` `maxrep` `` in place of either a distribution or upperbound.
- negated ranges are output as `neg(...)` since we typically
  don't want to generate an arbitrary value outside the specified range, you
  can implement a `neg` gfunc or fix it.
- if the tokenizer skips whitespace, you need to decide where to generate
  whitespace.  e.g. add spaces to keyword literals.
- recursion is unchecked, so if sampling the resulting grammar hangs,
  identify alternations where the recursive route is taken and make them less
  likely.


# setup

1. read [setup.sh](setup.sh)
2. run `setup.sh`

# usage
e.g. for json
``` bash
    ./g4toglf.py grammars-v4/json/JSON.g4 -o JSON.glf
```

When you have multiple g4 files, make sure the parser comes last:
```bash
    ./g4toglf.py grammars-v4/javascript/javascript/JavaScript{Lexer,Parser}.g4 -o javascript.glf
```


ANTLR grammars are for parsing and not generation, so certain constructs need
tweaking.

- in ANTLR, `X*` will recognize any sequence of `X`'s... to generate, we need
  a distribution on the number of repetitions.  g4toglf uses a parameter
  `` `maxrep` `` in place of either a distribution or upperbound.
- negated ranges are output as `neg(...)` since we typically
  don't want to generate an arbitrary value outside the specified range, you
  can implement a `neg` gfunc or fix it.
- other constructs get gfunc names as well: `eof` and `any`
- Antlr4 actions execute code during the parse.  g4toglf will pass the string
  contents of an action to a gfunc `action`.
- if the tokenizer skips whitespace, you need to decide where to generate
  whitespace.  e.g. add spaces to keyword literals.
- recursion is unchecked, so if sampling the resulting grammar hangs,
  identify alternations where the recursive route is taken and make them less
  likely.

Once you've got GLF that gramma can parse, you can experiment with
```bash
    ./tryitout.py my_new_grammar.glf
```
If the sample takes more than 5 seconds, every 5 seconds the StackWatcher
sideeffect will dump a count of rules on the generation stack.  You should
consider limiting the recursions that cause the most frequent rules.


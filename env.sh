D="$(cd "$(dirname "$(realpath ${BASH_SOURCE[0]})")" && cd -P "$(dirname "$SOURCE")" && pwd)"

export PYTHONPATH="${D}${PYTHONPATH:+:${PYTHONPATH}}"

alias g4toglf="python3 -c 'import gramma.converters.antlr4.g4toglf as m;m.main()'"
alias glf2cpp="python3 -c 'import gramma.samplers.generators.cpp as m;m.main()'"

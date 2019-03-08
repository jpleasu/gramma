#!/bin/bash

wget https://www.antlr.org/download/antlr-4.7.2-complete.jar
git clone https://github.com/antlr/grammars-v4

java -jar antlr-4.7.2-complete.jar -Dlanguage=Python2 -Xexact-output-dir  -o antlr4parser -package antlr4parser ./grammars-v4/antlr4/ANTLRv4Lexer.g4 
java -jar antlr-4.7.2-complete.jar -Dlanguage=Python2 -Xexact-output-dir  -o antlr4parser -package antlr4parser ./grammars-v4/antlr4/ANTLRv4LexerPythonTarget.g4 
java -jar antlr-4.7.2-complete.jar -Dlanguage=Python2 -Xexact-output-dir  -o antlr4parser -package antlr4parser -visitor ./grammars-v4/antlr4/ANTLRv4Parser.g4 

touch antlr4parser/__init__.py
cp grammars-v4/antlr4/LexerAdaptor.py antlr4parser/

echo ======
echo remember to install runtime, e.g.
echo \   pip2 install --user antlr4-python2-runtime
echo \   pip3 install --user antlr4-python3-runtime



#!/bin/bash

ANTLRVERSION=4.8
wget https://www.antlr.org/download/antlr-${ANTLRVERSION}-complete.jar
git clone https://github.com/antlr/grammars-v4

java -jar antlr-${ANTLRVERSION}-complete.jar -Dlanguage=Python3 -Xexact-output-dir  -o antlr4parser ./grammars-v4/antlr/antlr4/ANTLRv4Lexer.g4 
java -jar antlr-${ANTLRVERSION}-complete.jar -Dlanguage=Python3 -Xexact-output-dir  -o antlr4parser ./grammars-v4/antlr/antlr4/ANTLRv4LexerPythonTarget.g4 
java -jar antlr-${ANTLRVERSION}-complete.jar -Dlanguage=Python3 -Xexact-output-dir  -o antlr4parser -visitor ./grammars-v4/antlr/antlr4/ANTLRv4Parser.g4 

touch antlr4parser/__init__.py
cp grammars-v4/antlr/antlr4/LexerAdaptor.py antlr4parser/

echo ======
echo remember to install runtime, e.g.
echo \   pip3 install antlr4-python3-runtime


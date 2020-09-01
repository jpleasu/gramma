#!/bin/bash

ANTLR_VERSION=4.8

.PHONY:clean all

all: antlr4parser/__init__.py

antlr-$(ANTLR_VERSION)-complete.jar:
	curl -O https://www.antlr.org/download/antlr-$(ANTLR_VERSION)-complete.jar

grammars-v4:
	git clone https://github.com/antlr/grammars-v4

antlr4parser/__init__.py: antlr-$(ANTLR_VERSION)-complete.jar grammars-v4
	java -jar antlr-${ANTLR_VERSION}-complete.jar -Dlanguage=Python3 -package antlr4parser -Xexact-output-dir -o antlr4parser ./grammars-v4/antlr/antlr4/ANTLRv4Lexer.g4
	java -jar antlr-${ANTLR_VERSION}-complete.jar -Dlanguage=Python3 -package antlr4parser -Xexact-output-dir -o antlr4parser ./grammars-v4/antlr/antlr4/ANTLRv4LexerPythonTarget.g4
	java -jar antlr-${ANTLR_VERSION}-complete.jar -Dlanguage=Python3 -package antlr4parser -Xexact-output-dir -o antlr4parser -visitor ./grammars-v4/antlr/antlr4/ANTLRv4Parser.g4
	
	cp grammars-v4/antlr/antlr4/LexerAdaptor.py antlr4parser/
	touch antlr4parser/__init__.py


clean:
	rm -rf grammars-v4 antlr-*-complete.jar antlr4parser

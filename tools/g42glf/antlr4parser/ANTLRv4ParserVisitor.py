# Generated from ./grammars-v4/antlr4/ANTLRv4Parser.g4 by ANTLR 4.7.2
from antlr4 import *

# This class defines a complete generic visitor for a parse tree produced by ANTLRv4Parser.

class ANTLRv4ParserVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by ANTLRv4Parser#grammarSpec.
    def visitGrammarSpec(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#grammarDecl.
    def visitGrammarDecl(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#grammarType.
    def visitGrammarType(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#prequelConstruct.
    def visitPrequelConstruct(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#optionsSpec.
    def visitOptionsSpec(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#option.
    def visitOption(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#optionValue.
    def visitOptionValue(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#delegateGrammars.
    def visitDelegateGrammars(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#delegateGrammar.
    def visitDelegateGrammar(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#tokensSpec.
    def visitTokensSpec(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#channelsSpec.
    def visitChannelsSpec(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#idList.
    def visitIdList(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#action_.
    def visitAction_(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#actionScopeName.
    def visitActionScopeName(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#actionBlock.
    def visitActionBlock(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#argActionBlock.
    def visitArgActionBlock(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#modeSpec.
    def visitModeSpec(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#rules.
    def visitRules(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#ruleSpec.
    def visitRuleSpec(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#parserRuleSpec.
    def visitParserRuleSpec(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#exceptionGroup.
    def visitExceptionGroup(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#exceptionHandler.
    def visitExceptionHandler(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#finallyClause.
    def visitFinallyClause(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#rulePrequel.
    def visitRulePrequel(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#ruleReturns.
    def visitRuleReturns(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#throwsSpec.
    def visitThrowsSpec(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#localsSpec.
    def visitLocalsSpec(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#ruleAction.
    def visitRuleAction(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#ruleModifiers.
    def visitRuleModifiers(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#ruleModifier.
    def visitRuleModifier(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#ruleBlock.
    def visitRuleBlock(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#ruleAltList.
    def visitRuleAltList(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#labeledAlt.
    def visitLabeledAlt(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#lexerRuleSpec.
    def visitLexerRuleSpec(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#lexerRuleBlock.
    def visitLexerRuleBlock(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#lexerAltList.
    def visitLexerAltList(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#lexerAlt.
    def visitLexerAlt(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#lexerElements.
    def visitLexerElements(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#lexerElement.
    def visitLexerElement(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#labeledLexerElement.
    def visitLabeledLexerElement(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#lexerBlock.
    def visitLexerBlock(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#lexerCommands.
    def visitLexerCommands(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#lexerCommand.
    def visitLexerCommand(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#lexerCommandName.
    def visitLexerCommandName(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#lexerCommandExpr.
    def visitLexerCommandExpr(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#altList.
    def visitAltList(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#alternative.
    def visitAlternative(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#element.
    def visitElement(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#labeledElement.
    def visitLabeledElement(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#ebnf.
    def visitEbnf(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#blockSuffix.
    def visitBlockSuffix(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#ebnfSuffix.
    def visitEbnfSuffix(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#lexerAtom.
    def visitLexerAtom(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#atom.
    def visitAtom(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#notSet.
    def visitNotSet(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#blockSet.
    def visitBlockSet(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#setElement.
    def visitSetElement(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#block.
    def visitBlock(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#ruleref.
    def visitRuleref(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#characterRange.
    def visitCharacterRange(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#terminal.
    def visitTerminal(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#elementOptions.
    def visitElementOptions(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#elementOption.
    def visitElementOption(self, ctx):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ANTLRv4Parser#identifier.
    def visitIdentifier(self, ctx):
        return self.visitChildren(ctx)



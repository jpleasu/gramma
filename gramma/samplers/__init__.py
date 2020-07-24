__all__ = [
    'GrammaInterpereterBase',
]

from typing import Union, IO, Final

from ..parser import GrammaGrammar


class RandomAPI:
    """a proxy to numpy.random"""
    def __init__(self):
        pass


class GrammaInterpereterBase:
    grammar: Final[GrammaGrammar]
    random: RandomAPI

    def __init__(self, grammar: Union[IO[str], str, GrammaGrammar]):
        """
        grammar is either a GrammarGrammar object, a string containing GLF, or a file handle to a GLF file.
        """
        self.grammar = GrammaGrammar.of(grammar)

        # XXX: bind methods of this class to GFunc objects
        # XXX: bind GCode to closures on this class context

from __future__ import annotations

__all__ = [
    'GrammaInterpreterSamplerBase',
    'gfunc',
]


import logging
from types import CodeType
from typing import Union, IO, Final, Dict, Any, List, Callable, TypeVar, Protocol, runtime_checkable, Optional

from ..parser import GrammaGrammar, GFunc, GCode

log = logging.getLogger('gramma.samplers')


class RandomAPI:
    """a proxy to numpy.random"""

    def __init__(self):
        pass


class GrammaSamplerError(Exception):
    pass


class GFuncWrap(object):
    __slots__ = 'func', 'fname'
    fname: str
    func: Callable

    def __init__(self, func, fname):
        self.func = func
        self.fname = fname

    def __call__(self, *args, **kw):
        return self.func(*args, **kw)

    def __str__(self):
        return f'gfunc {self.fname}'

    def copy(self):
        return GFuncWrap(self.func, self.fname)


def gfunc(*args, **kw):
    """
    decorator sampler methods to indicate implementation of a GFunc element
    """

    def _decorate(func, **kw):
        fname = kw.pop('fname', func.__name__)
        return GFuncWrap(func, fname)

    if len(args) == 0 or not callable(args[0]):
        return lambda func: _decorate(func, *args, **kw)

    return _decorate(args[0], **kw)


gcode_globals: Final[Dict[str, Any]] = {}


class GCodeWrap:
    __slots__ = 'code', 'compiled'
    code: Final[GCode]
    compiled: Final[CodeType]

    def __init__(self, code: GCode):
        self.code = code
        self.compiled = compile(code.expr, '<GCode>', 'eval')

    def __call__(self, sampler: GrammaInterpreterSamplerBase):
        return eval(self.compiled, gcode_globals, sampler.__dict__)


T = TypeVar('T')


@runtime_checkable
class SupportsAdd(Protocol[T]):
    def __add__(self: T, other: T) -> T:
        ...


class Strang:
    __slots__ = 's', 'val'

    s: str
    val: Optional[SupportsAdd]

    def __init__(self, s: str, val: Optional[SupportsAdd] = None):
        self.s = s
        self.val = val

    def denote(self, val):
        self.val = val

    def __add__(self, other: Strang):
        if other.val is None:
            return Strang(self.s + other.s, self.val)
        elif self.val is None:
            return Strang(self.s + other.s, other.val)
        else:
            return Strang(self.s + other.s, self.val + other.val)

    def __iadd__(self, other: Strang):
        self.s += other.s
        if other.val is not None:
            if self.val is None:
                self.val = other.val
            else:
                self.val += other.val
        return self

    def __str__(self):
        return self.s


class GrammaInterpreterSamplerBase:
    grammar: Final[GrammaGrammar]
    random: RandomAPI
    gfuncmap: Dict[GFunc, GFuncWrap]
    gcodemap: Dict[GCode, GCodeWrap]

    def _(self, return_=None, **kw):
        """
        a convenience for assigning to the sampler from gcode
            e.g.
                `_(x=5)` returns None
                `_(x, x=5)` returns the current value of sampler.x and updates x
        """
        self.__dict__.update(kw)
        return return_

    def __init__(self, grammar: Union[IO[str], str, GrammaGrammar]):
        """
        grammar is either a GrammaGrammar object, a string containing GLF, or a file handle to a GLF file.
        """
        self.grammar = GrammaGrammar.of(grammar)
        self.gfuncmap = {}
        self.gcodemap = {}
        gfuncrefs: Dict[str, List[GFunc]] = {}
        gcode: List[GCode] = []

        # get GCode and GFunc refs from AST
        for ge in self.grammar.walk():
            if isinstance(ge, GFunc):
                gfuncrefs.setdefault(ge.fname, []).append(ge)
            elif isinstance(ge, GCode):
                gcode.append(ge)
            for gc in ge.get_code():
                gcode.append(gc)
        for code in gcode:
            self.gcodemap[code] = GCodeWrap(code)

        # find gfunc implementations
        for name in dir(self):
            if name.startswith('__'):
                continue
            value = getattr(self, name)
            if isinstance(value, GFuncWrap):
                for ge in gfuncrefs.get(value.fname, []):
                    self.gfuncmap[ge] = value
                del gfuncrefs[value.fname]

        if len(gfuncrefs) > 0:
            for ge_list in gfuncrefs.values():
                for ge in ge_list:
                    log.error(f'no implementation in {self.__class__.__name__} for {ge.fname}')
            raise GrammaSamplerError('sampler is missing gfunc implementations')

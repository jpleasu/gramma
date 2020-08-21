from functools import reduce
from types import CodeType
from typing import Union, IO, Final, Dict, Any, Callable, Optional, List, TypeVar, Protocol, Type

# numpy doesn't have type hints yet and data-science-type is missing numpy.random.Generator
import numpy as np  # type: ignore

from . import log
from ..parser import GrammaGrammar, GFuncRef, GCode, GDFuncRef, GCat, GAlt, GTok, GRuleRef, GRep, GExpr, GVarRef, \
    GChooseIn, GDenoted
from ..util import DictStack

T = TypeVar('T')


class RandomAPI:
    """a proxy to numpy.random"""
    __slots__ = 'generator',

    generator: np.random.Generator

    def __init__(self, seed=None):
        self.generator = np.random.Generator(np.random.MT19937(np.random.SeedSequence(seed)))

    def choice(self, choices: List[T], weights=Union[None, List[Union[int, float]], np.ndarray]) -> T:
        if weights is None:
            weights = np.ones(len(choices))
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)
        p = weights / weights.sum()
        return self.generator.choice(choices, p=p)

    def seed(self, v: int) -> None:
        self.generator = np.random.Generator(np.random.MT19937(np.random.SeedSequence(v)))

    def integers(self, lo: int, hi: int) -> int:
        return self.generator.integers(lo, hi)

    def geometric(self, p: float) -> float:
        return self.generator.geometric(p)


class GrammaSamplerError(Exception):
    pass


class GFuncWrap:
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
        return self.__class__(self.func, self.fname)


class GDFuncWrap(GFuncWrap):
    pass


def make_decorator(wrapper_class: Type):
    def _decorate(func, **kw):
        fname = kw.pop('fname', func.__name__)
        return wrapper_class(func, fname)

    def decorator(*args, **kw):
        f"""decorator for sampler methods to indicate {wrapper_class.__class__.__name__} implementation"""

        if len(args) == 0 or not callable(args[0]):
            return lambda func: _decorate(func, *args, **kw)

        return _decorate(args[0], **kw)

    return decorator


gfunc = make_decorator(GFuncWrap)
gdfunc = make_decorator(GDFuncWrap)

gcode_globals: Final[Dict[str, Any]] = {}


class GCodeWrap:
    __slots__ = 'code', 'compiled'
    code: Final[GCode]
    compiled: Final[CodeType]

    def __init__(self, code: GCode):
        self.code = code
        self.compiled = compile(code.expr, '<GCode>', 'eval')

    def __call__(self, sampler: 'GrammaInterpreter'):
        return eval(self.compiled, gcode_globals, sampler.__dict__)


class Denotation(Protocol):
    ...


class Sample:
    __slots__ = 's', 'd'

    s: str
    d: Optional[Denotation]

    def __init__(self, s: str, val: Denotation = None):
        self.s = s
        self.d = val

    def denote(self, val: Denotation) -> 'Sample':
        return Sample(self.s, val)

    def cat(self, other: 'Sample') -> 'Sample':
        return Sample(self.s + other.s)

    @staticmethod
    def get_unit() -> 'Sample':
        return Sample('')

    def __str__(self):
        return self.s


class SamplerMixinInterface(Protocol):
    random: RandomAPI
    grammar: GrammaGrammar
    gfuncmap: Dict[GFuncRef, GFuncWrap]
    gdfuncmap: Dict[GDFuncRef, GDFuncWrap]
    gcodemap: Dict[GCode, GCodeWrap]

    @staticmethod
    def create_sample(*args, **kwargs) -> Sample:
        ...

    def sample(self, start: Optional[GExpr]) -> Sample:
        ...

    def evaluate_denotation(self, ge: GExpr) -> Denotation:
        ...

    def exec(self, gc: GCode) -> Any:
        ...

    def eval_num(self, ge: Union[GTok, GCode]) -> Union[int, float]:
        ...

    def eval_int(self, ge: Union[GTok, GCode]) -> int:
        ...


class GCodeHelpersSamplerMixin(SamplerMixinInterface):
    """
    methods intended for use by GCode
    """

    def _(self, return_=None, **kw):
        """
        a convenience for assigning to the sampler from gcode
            e.g.
                `_(x=5)` returns None
                `_(x, x=5)` returns the current value of sampler.x and updates x
                `_(x, x=5)` same, but explicit
                `_(x=5, _return=x)` return the value of x after update
        """
        self.__dict__.update(kw)
        return return_


# noinspection PyPep8Naming
class OperatorsImplementationSamplerMixin(SamplerMixinInterface):
    vars: DictStack[str, Sample]

    def __init__(self):
        self.vars = DictStack()

    def sample_GTok(self, ge: GTok) -> Sample:
        return self.create_sample(ge.as_str())

    def sample_GCat(self, ge: GCat) -> Sample:
        samples = [self.sample(c) for c in ge.children]
        return reduce(Sample.cat, samples, Sample.get_unit())

    def sample_GAlt(self, ge: GAlt) -> Sample:
        weights = [self.eval_num(c) for c in ge.weights]
        return self.sample(self.random.choice(ge.children, weights))

    def sample_GRep(self, ge: GRep) -> Sample:
        lo: Union[int, None]
        hi: Union[int, None]
        if ge.lo is None:
            lo = None
        else:
            lo = self.eval_int(ge.lo)
        if ge.hi is None:
            hi = None
        else:
            hi = self.eval_int(ge.hi)

        d = ge.dist.name
        n: int
        if lo == hi and lo is not None:
            n = lo
        elif d.startswith('unif'):
            if lo is None:
                lo = 0
            if hi is None:
                hi = 2 ** 32
            n = self.random.integers(lo, hi + 1)
        elif d.startswith('geom'):
            # geom(n) samples have a mean of n
            n = int(.5 + self.random.geometric(1 / (ge.dist.args[0].as_int() + 1)))

        else:
            raise GrammaSamplerError(f"sampler has no handler for repetition distrituion {d}")

        # truncate
        if lo is not None:
            n = max(lo, n)
        if hi is not None:
            n = min(n, hi)

        if n == 0:
            return self.create_sample('')

        n -= 1
        s = self.sample(ge.child)
        while n > 0:
            s = s.cat(self.sample(ge.child))
            n -= 1
        return s

    def sample_GRuleRef(self, ge: GRuleRef) -> Sample:
        rule = self.grammar.ruledefs[ge.rname]
        args = [self.sample(c) for c in ge.rargs]
        with self.vars.context(dict(zip(rule.params, args))):
            samp = self.sample(rule.rhs)
        return samp

    def sample_GChooseIn(self, ge: GChooseIn) -> Sample:
        dists = [self.sample(c) for c in ge.dists]
        with self.vars.context(dict(zip(ge.vnames, dists))):
            return self.sample(ge.child)

    def sample_GVarRef(self, ge: GVarRef) -> Sample:
        s = self.vars.get(ge.vname)
        if s is None:
            raise GrammaSamplerError(f'undefined variable "{ge.vname}" ')
        return s

    def sample_GFuncRef(self, ge: GFuncRef) -> Sample:
        gfw = self.gfuncmap[ge]
        fargs = [self.sample(c) for c in ge.fargs]
        return gfw.func(self, *fargs)

    def sample_GDenoted(self, ge: GDenoted) -> Sample:
        s = self.sample(ge.left)
        val = self.evaluate_denotation(ge.right)
        return s.denote(val)


# noinspection PyPep8Naming,PyMethodMayBeStatic
class DenotationImplementationSamplerMixin(SamplerMixinInterface):
    def evaluate_denotation_GTok(self, ge: GTok) -> Denotation:
        return ge.as_native()

    def evaluate_denotation_GDFuncRef(self, ge: GDFuncRef) -> Denotation:
        gdfw = self.gdfuncmap[ge]
        fargs = [self.sample(c) for c in ge.fargs]
        return gdfw.func(self, *fargs)

    def evaluate_denotation_GCode(self, ge: GCode) -> Denotation:
        return self.exec(ge)


class GrammaInterpreter(DenotationImplementationSamplerMixin, OperatorsImplementationSamplerMixin,
                        GCodeHelpersSamplerMixin):
    """
    A sampler that interprets GLF on the function stack.  It's slow, so only use it to prototype and debug a grammar.
    """
    __slots__ = 'grammar', 'random', 'gfuncmap', 'gdfuncmap', 'gcodemap'
    grammar: GrammaGrammar
    random: RandomAPI
    gfuncmap: Dict[GFuncRef, GFuncWrap]
    gdfuncmap: Dict[GDFuncRef, GDFuncWrap]
    gcodemap: Dict[GCode, GCodeWrap]

    @staticmethod
    def create_sample(*args, **kwargs) -> Sample:
        return Sample(*args, **kwargs)

    def __init__(self, grammar: Union[IO[str], str, GrammaGrammar]):
        """
        grammar is either a GrammaGrammar object, a string containing GLF, or a file handle to a GLF file.
        """
        OperatorsImplementationSamplerMixin.__init__(self)
        GCodeHelpersSamplerMixin.__init__(self)

        self.grammar = GrammaGrammar.of(grammar)
        self.gfuncmap = {}
        self.gdfuncmap = {}
        self.gcodemap = {}
        self.random = RandomAPI()

        # create method lookups
        gdfuncs: Dict[str, GDFuncWrap] = {}
        gfuncs: Dict[str, GFuncWrap] = {}

        for name in dir(self):
            if name.startswith('__'):
                continue
            val = getattr(self, name)
            if isinstance(val, GDFuncWrap):
                gdfuncs[val.fname] = val
            elif isinstance(val, GFuncWrap):
                gfuncs[val.fname] = val

        # wire GCode, GFunc, and GDFunc refs from AST to Wraps
        missing: List[Union[GFuncRef, GDFuncRef]] = []
        for ge in self.grammar.walk():
            if isinstance(ge, GDFuncRef):
                wd = gdfuncs.get(ge.fname)
                if wd is None:
                    missing.append(ge)
                else:
                    self.gdfuncmap[ge] = wd
            elif isinstance(ge, GFuncRef):
                w = gfuncs.get(ge.fname)
                if w is None:
                    missing.append(ge)
                else:
                    self.gfuncmap[ge] = w
            elif isinstance(ge, GCode):
                self.gcodemap[ge] = GCodeWrap(ge)
            for gc in ge.get_code():
                self.gcodemap[gc] = GCodeWrap(gc)

        if len(missing) > 0:
            for ge in missing:
                log.error(f'no implementation in {self.__class__.__name__} for {ge.__class__.__name__} "{ge.fname}"')
            raise GrammaSamplerError('sampler is missing function implementations')

    def _(self, return_=None, **kw):
        """
        a convenience for assigning to the sampler from gcode
            e.g.
                `_(x=5)` returns None
                `_(x, x=5)` returns the current value of sampler.x and updates x
        """
        self.__dict__.update(kw)
        return return_

    def exec(self, gc: GCode) -> Any:
        return self.gcodemap[gc](self)

    def eval_int(self, ge: Union[GTok, GCode]) -> int:
        return ge.as_int() if isinstance(ge, GTok) else self.exec(ge)

    def eval_num(self, ge: Union[GTok, GCode]) -> Union[float, int]:
        return ge.as_num() if isinstance(ge, GTok) else self.exec(ge)

    def sample(self, start=None) -> Sample:
        """
        draw a sample from the distribution defined by grammar and its sampler methods
        """
        if start is None:
            start = self.grammar.ruledefs['start'].rhs
        handler_name = 'sample_' + start.__class__.__name__
        m = getattr(self, handler_name, None)
        if m is None:
            log.error(f'missing handler in {self.__class__.__name__}, expected method {handler_name}')
            raise GrammaSamplerError('sampler is missing sample_* method')
        return m(start)

    def evaluate_denotation(self, ge: GExpr) -> Denotation:
        handler_name = 'evaluate_denotation_' + ge.__class__.__name__
        m = getattr(self, handler_name, None)
        if m is None:
            log.error(f'missing handler in {self.__class__.__name__}, expected method {handler_name}')
            raise GrammaSamplerError('sampler is missing evaluate_denotation_* method')
        return m(ge)

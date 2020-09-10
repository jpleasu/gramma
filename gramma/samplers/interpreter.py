from functools import reduce
from types import CodeType
from typing import Union, IO, Final, Dict, Any, Callable, List, TypeVar, Protocol, Type, Generator, cast, Optional, \
    Generic

import numpy as np

from ..parser import GrammaGrammar, GFuncRef, GCode, GDFuncRef, GCat, GAlt, GTok, GRuleRef, GRep, GExpr, GVarRef, \
    GChooseIn, GDenoted, GRange, GTern
from ..util import DictStack

import logging

log = logging.getLogger('gramma.samplers.interpreter')

T = TypeVar('T')

# the value used when the repetition isn't given an upper limit
REP_HIGH = 2 ** 10


class RandomAPI:
    """a proxy to numpy.random"""
    __slots__ = 'generator',

    generator: np.random.Generator

    def __init__(self, seed=None):
        self.generator = np.random.Generator(np.random.MT19937(np.random.SeedSequence(seed)))

    def seed(self, v: int) -> None:
        self.generator = np.random.Generator(np.random.MT19937(np.random.SeedSequence(v)))

    def choice(self, choices: List[T], weights: Union[None, List[Union[int, float]], np.ndarray] = None) -> T:
        if weights is None:
            weights = np.ones(len(choices))
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights)
        p = weights / weights.sum()
        return cast(T, self.generator.choice(choices, p=p))

    def integers(self, lo: int, hi: int) -> int:
        return cast(int, self.generator.integers(lo, hi))

    def geometric(self, p: float) -> int:
        return cast(int, self.generator.geometric(p))

    def normal(self, mean: float, std: float) -> float:
        return cast(float, self.generator.normal(mean, std))

    def binomial(self, n: int, p: float) -> int:
        return cast(int, self.generator.binomial(n, p))


DenotationType = TypeVar('DenotationType', contravariant=True)


class Sample(Generic[DenotationType]):
    __slots__ = 's', 'd'

    s: str
    d: Optional[DenotationType]

    def __init__(self, s: str, val: Optional[DenotationType] = None):
        self.s = s
        self.d = val

    def __str__(self) -> str:
        return self.s


SampleType = TypeVar('SampleType', bound=Sample[Any])

gcode_globals: Final[Dict[str, Any]] = {'Sample': Sample}


class GrammaSamplerError(Exception):
    pass


class FuncType(Protocol):  # pragma: no cover
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...


class GFuncWrap:
    __slots__ = 'func', 'fname', 'lazy'
    fname: str
    func: FuncType
    lazy: bool

    def __init__(self, func, fname, lazy=False):
        self.func = func
        self.fname = fname
        self.lazy = lazy


class GDFuncWrap(GFuncWrap):
    pass


U = TypeVar('U', bound=Union[Type[GDFuncWrap], Type[GFuncWrap]])


def make_decorator(wrapper_class: U) -> Callable[..., Any]:
    def _decorate(func, **kw):
        fname = kw.pop('fname', func.__name__)
        return wrapper_class(func, fname, **kw)

    def decorator(*args, **kw):
        f"""decorator for sampler methods to indicate {wrapper_class.__class__.__name__} implementation"""

        if len(args) == 0 or not callable(args[0]):
            return lambda func: _decorate(func, *args, **kw)

        return _decorate(args[0], **kw)

    return decorator


gfunc = make_decorator(GFuncWrap)
gdfunc = make_decorator(GDFuncWrap)


class GCodeWrap:
    __slots__ = 'code', 'compiled'
    code: Final[GCode]
    compiled: Final[CodeType]

    def __init__(self, code: GCode):
        self.code = code
        self.compiled = compile(code.expr, '<GCode>', 'eval')


class SamplerInterface(Protocol[SampleType, DenotationType]):  # pragma: no cover
    random: RandomAPI
    grammar: GrammaGrammar
    gfuncmap: Dict[GFuncRef, GFuncWrap]
    coro_gfuncmap: Dict[GFuncRef, GFuncWrap]
    gdfuncmap: Dict[GDFuncRef, GDFuncWrap]
    coro_gdfuncmap: Dict[GDFuncRef, GDFuncWrap]
    gcodemap: Dict[GCode, GCodeWrap]

    def create_sample(self, *args: Any, **kwargs: Any) -> SampleType:
        ...

    def get_unit(self) -> SampleType:
        ...

    def cat(self, a: SampleType, b: SampleType) -> SampleType:
        ...

    def denote(self, a: SampleType, b: DenotationType) -> SampleType:
        ...

    def exec(self, gc: GCode) -> Any:
        ...

    def eval_num(self, ge: Union[GTok, GCode]) -> Union[int, float]:
        ...

    def eval_int(self, ge: Union[GTok, GCode]) -> int:
        ...


# noinspection PyPep8Naming,PyMethodMayBeStatic
class OperatorsImplementationSamplerMixin(SamplerInterface[SampleType, DenotationType]):
    __slots__ = 'vars',
    vars: DictStack[str, SampleType]

    def __init__(self):
        self.vars = DictStack()

    def sample(self, ge: GExpr) -> SampleType:
        handler_name = 'sample_' + ge.__class__.__name__
        m = cast(Optional[Callable[[GExpr], SampleType]], getattr(self, handler_name, None))
        if m is None:  # pragma: no cover
            msg = f'missing handler in {self.__class__.__name__}: {handler_name}'
            log.error(msg)
            raise GrammaSamplerError(msg)
        return m(ge)

    def sample_GTok(self, ge: GTok) -> SampleType:
        return self.create_sample(ge.as_str())

    def sample_GCat(self, ge: GCat) -> SampleType:
        samples = [self.sample(c) for c in ge.children]
        return reduce(self.cat, samples, self.get_unit())

    def sample_GAlt(self, ge: GAlt) -> SampleType:
        weights = [self.eval_num(c) for c in ge.weights]
        return self.sample(self.random.choice(ge.children, weights))

    def sample_GTern(self, ge: GTern) -> SampleType:
        if self.exec(ge.code):
            return self.sample(ge.children[0])
        else:
            return self.sample(ge.children[1])

    def sample_GRep(self, ge: GRep) -> SampleType:
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

        dist = ge.dist
        n: int
        if lo == hi and lo is not None:
            n = lo
        elif dist.name.startswith('unif'):
            if lo is None:
                lo = 0
            if hi is None:
                hi = REP_HIGH
            n = self.random.integers(lo, hi + 1)
        elif dist.name.startswith('geom'):
            n = self.random.geometric(1 / (dist.args[0].as_int() + 1)) - 1
        elif dist.name.startswith('norm'):
            n = int(.5 + self.random.normal(dist.args[0].as_float(), dist.args[1].as_float()))
        elif dist.name.startswith('binom'):
            n = self.random.binomial(dist.args[0].as_int(), dist.args[1].as_float())
        elif dist.name == 'choose' or dist.name == 'choice':
            n = self.random.choice([a.as_int() for a in dist.args])
        else:
            raise GrammaSamplerError(f"sampler has no handler for repetition distribution {dist.name}")

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
            s = self.cat(s, self.sample(ge.child))
            n -= 1
        return s

    def sample_GRange(self, ge: GRange) -> SampleType:
        n = sum(c for b, c in ge.pairs)
        i = self.random.integers(0, n)
        for b, c in ge.pairs:
            if i < c:
                return self.create_sample(chr(b + i))
            i -= c
        raise GrammaSamplerError('range exceeded')  # pragma: no cover

    def sample_GRuleRef(self, ge: GRuleRef) -> SampleType:
        rule = self.grammar.ruledefs[ge.rname]
        args = [self.sample(c) for c in ge.rargs]
        if len(rule.params) != len(args):
            raise GrammaSamplerError(f'rule {ge.rname}/{len(rule.params)} called with {len(args)} argument(s)')
        with self.vars.context(dict(zip(rule.params, args))):
            samp = self.sample(rule.rhs)
        return samp

    def sample_GChooseIn(self, ge: GChooseIn) -> SampleType:
        dists = [self.sample(c) for c in ge.dists]
        with self.vars.context(dict(zip(ge.vnames, dists))):
            return self.sample(ge.child)

    def sample_GCode(self, ge: GCode) -> SampleType:
        v = self.exec(ge)
        if isinstance(v, Sample):
            return cast(SampleType, v)
        else:
            return self.create_sample(str(v))

    def sample_GVarRef(self, ge: GVarRef) -> SampleType:
        s = self.vars.get(ge.vname)
        if s is None:  # pragma: no cover
            raise GrammaSamplerError(f'undefined variable "{ge.vname}" ')
        return s

    def sample_GFuncRef(self, ge: GFuncRef) -> SampleType:
        gfw = self.gfuncmap[ge]
        func = cast(Callable[..., SampleType], gfw.func)
        if gfw.lazy:
            fargs = ge.fargs
        else:
            fargs = [self.sample(c) for c in ge.fargs]
        return func(self, *fargs)

    def sample_GDenoted(self, ge: GDenoted) -> SampleType:
        s = self.sample(ge.left)
        val = self.evaluate_denotation(ge.right)
        return self.denote(s, val)

    def evaluate_denotation(self, ge: GExpr) -> Any:
        handler_name = 'evaluate_denotation_' + ge.__class__.__name__
        m = getattr(self, handler_name, None)
        if m is None:  # pragma: no cover
            msg = f'missing handler in {self.__class__.__name__}: {handler_name}'
            log.error(msg)
            raise GrammaSamplerError(msg)
        return m(ge)

    evaluate_denotation_GVarRef = sample_GVarRef

    def evaluate_denotation_GTok(self, ge: GTok) -> Any:
        return ge.as_native()

    def evaluate_denotation_GDFuncRef(self, ge: GDFuncRef) -> Any:
        gdfw = self.gdfuncmap[ge]
        fargs = [self.evaluate_denotation(c) for c in ge.fargs]
        return gdfw.func(self, *fargs)

    def evaluate_denotation_GCode(self, ge: GCode) -> Any:
        return self.exec(ge)

    #######################################################
    # the following "coroutine" API consists of stack machines,
    # coro_sample and coro_evaluate_denotation and rewrites of
    # the sample_* methods.
    #######################################################

    def coro_sample(self, ge: GExpr) -> SampleType:
        stack: List[Generator[Union[GExpr, SampleType], SampleType, None]] = []

        while True:
            handler_name = 'coro_sample_' + ge.__class__.__name__
            m = getattr(self, handler_name, None)
            if m is None:
                msg = f'missing handler in {self.__class__.__name__}: {handler_name}'
                log.error(msg)
                raise GrammaSamplerError(msg)
            coro = m(ge)
            x: Union[SampleType, GExpr] = next(coro)
            while isinstance(x, Sample):
                if len(stack) == 0:
                    return cast(SampleType, x)
                coro = stack.pop()
                x = coro.send(x)
            stack.append(coro)
            ge = cast(GExpr, x)

    def coro_evaluate_denotation(self, ge: GExpr) -> Any:
        stack: List[Generator[Union[GExpr, Any], Any, None]] = []

        while True:
            handler_name = 'coro_evaluate_denotation_' + ge.__class__.__name__
            m = getattr(self, handler_name, None)
            if m is None:  # pragma: no cover
                msg = f'missing handler in {self.__class__.__name__}: {handler_name}'
                log.error(msg)
                raise GrammaSamplerError(msg)
            coro = m(ge)
            x: Union[Any, GExpr] = next(coro)
            while not isinstance(x, GExpr):
                if len(stack) == 0:
                    return x
                coro = stack.pop()
                x = coro.send(x)
            stack.append(coro)
            ge = x

    ##################

    def coro_sample_GTok(self, ge: GTok) -> Generator[Union[GExpr, SampleType], SampleType, None]:
        yield self.create_sample(ge.as_str())

    def coro_sample_GCat(self, ge: GCat) -> Generator[Union[GExpr, SampleType], SampleType, None]:
        samples = []
        for c in ge.children:
            samples.append((yield c))
        yield reduce(self.cat, samples, self.get_unit())

    def coro_sample_GAlt(self, ge: GAlt) -> Generator[Union[GExpr, SampleType], SampleType, None]:
        weights = [self.eval_num(c) for c in ge.weights]
        yield (yield self.random.choice(ge.children, weights))

    def coro_sample_GTern(self, ge: GTern) -> Generator[Union[GExpr, SampleType], SampleType, None]:
        if self.exec(ge.code):
            yield (yield ge.children[0])
        else:
            yield (yield ge.children[1])

    def coro_sample_GRep(self, ge: GRep) -> Generator[Union[GExpr, SampleType], SampleType, None]:
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
        dist = ge.dist
        n: int
        if lo == hi and lo is not None:
            n = lo
        elif dist.name.startswith('unif'):
            if lo is None:
                lo = 0
            if hi is None:
                hi = REP_HIGH
            n = self.random.integers(lo, hi + 1)
        elif dist.name.startswith('geom'):
            n = self.random.geometric(1 / (dist.args[0].as_int() + 1)) - 1
        elif dist.name.startswith('norm'):
            n = int(0.5 + self.random.normal(dist.args[0].as_float(), dist.args[1].as_float()))
        elif dist.name.startswith('binom'):
            n = self.random.binomial(dist.args[0].as_int(), dist.args[1].as_float())
        elif dist.name == 'choose' or dist.name == 'choice':
            n = self.random.choice([a.as_int() for a in dist.args])
        else:
            raise GrammaSamplerError(f'sampler has no handler for repetition distribution {dist.name}')
        if lo is not None:
            n = max(lo, n)
        if hi is not None:
            n = min(n, hi)
        if n == 0:
            yield self.create_sample('')
        n -= 1
        s = yield ge.child
        while n > 0:
            s = self.cat(s, (yield ge.child))
            n -= 1
        yield s

    def coro_sample_GRange(self, ge: GRange) -> Generator[Union[GExpr, SampleType], SampleType, None]:
        n = sum(c for b, c in ge.pairs)
        i = self.random.integers(0, n)
        for b, c in ge.pairs:
            if i < c:
                yield self.create_sample(chr(b + i))
            i -= c
        raise GrammaSamplerError('range exceeded')  # pragma: no cover

    def coro_sample_GRuleRef(self, ge: GRuleRef) -> Generator[Union[GExpr, SampleType], SampleType, None]:
        rule = self.grammar.ruledefs[ge.rname]
        args = []
        for c in ge.rargs:
            args.append((yield c))
        if len(rule.params) != len(args):
            raise GrammaSamplerError(
                f'rule {ge.rname}/{len(rule.params)} called with {len(args)} argument(s)'
            )
        with self.vars.context(dict(zip(rule.params, args))):
            samp = yield rule.rhs
        yield samp

    def coro_sample_GChooseIn(self, ge: GChooseIn) -> Generator[Union[GExpr, SampleType], SampleType, None]:
        dists = []
        for c in ge.dists:
            dists.append((yield c))
        with self.vars.context(dict(zip(ge.vnames, dists))):
            yield (yield ge.child)

    def coro_sample_GCode(self, ge: GCode) -> Generator[Union[GExpr, SampleType], SampleType, None]:
        v = self.exec(ge)
        if isinstance(v, Sample):
            yield cast(SampleType, v)
        else:
            yield self.create_sample(str(v))

    def coro_sample_GVarRef(self, ge: GVarRef) -> Generator[Union[GExpr, SampleType], SampleType, None]:
        s = self.vars.get(ge.vname)
        if s is None:  # pragma: no cover
            raise GrammaSamplerError(f'undefined variable "{ge.vname}" ')
        yield s

    def coro_sample_GFuncRef(self, ge: GFuncRef) -> Generator[Union[GExpr, SampleType], SampleType, None]:
        gfw = self.coro_gfuncmap.get(ge, None)
        coro = True
        if gfw is None:
            gfw = self.gfuncmap[ge]
            coro = False

        if gfw.lazy:
            fargs = ge.fargs
        else:
            fargs = []
            for c in ge.fargs:
                fargs.append((yield c))

        if coro:
            yield from gfw.func(self, *fargs)
        else:
            yield gfw.func(self, *fargs)

    def coro_sample_GDenoted(self, ge: GDenoted) -> Generator[Union[GExpr, SampleType], SampleType, None]:
        s = yield ge.left
        val = self.coro_evaluate_denotation(ge.right)
        yield self.denote(s, val)

    coro_evaluate_denotation_GVarRef = coro_sample_GVarRef

    def coro_evaluate_denotation_GTok(self, ge: GTok) -> Generator[Union[GExpr, Any], Any, None]:
        yield ge.as_native()

    def coro_evaluate_denotation_GDFuncRef(self, ge: GDFuncRef) -> Generator[Union[GExpr, Any], Any, None]:
        gdfw = self.coro_gdfuncmap.get(ge, None)
        coro = True
        if gdfw is None:
            gdfw = self.gdfuncmap[ge]
            coro = False

        fargs = []
        for c in ge.fargs:
            fargs.append((yield c))
        if coro:
            yield from gdfw.func(self, *fargs)
        else:
            yield gdfw.func(self, *fargs)

    def coro_evaluate_denotation_GCode(self, ge: GCode) -> Generator[Union[GExpr, Any], Any, None]:
        yield self.exec(ge)


class GrammaInterpreter(OperatorsImplementationSamplerMixin[SampleType, DenotationType],
                        SamplerInterface[SampleType, DenotationType]):
    """
    A sampler that interprets GLF on the function stack.  It's slow, so only use it to prototype and debug a grammar.
    """
    __slots__ = 'grammar', 'random', 'gfuncmap', 'coro_gfuncmap', 'gdfuncmap', 'coro_gdfuncmap', 'gcodemap'
    grammar: GrammaGrammar
    random: RandomAPI
    gfuncmap: Dict[GFuncRef, GFuncWrap]
    coro_gfuncmap: Dict[GFuncRef, GFuncWrap]
    gdfuncmap: Dict[GDFuncRef, GDFuncWrap]
    coro_gdfuncmap: Dict[GDFuncRef, GDFuncWrap]
    gcodemap: Dict[GCode, GCodeWrap]

    @staticmethod
    def create_sample(*args: Any, **kwargs: Any) -> SampleType:
        """Override with SampleType constructor"""
        return cast(SampleType, Sample(*args, **kwargs))

    @staticmethod
    def get_unit() -> SampleType:
        """Override with SampleType constructor"""
        return cast(SampleType, Sample(''))

    @staticmethod
    def cat(a: SampleType, b: SampleType) -> SampleType:
        """Override with SampleType cat"""
        return cast(SampleType, Sample(a.s + b.s))

    def denote(self, a: SampleType, b: DenotationType) -> SampleType:
        """Override with SampleType denotation"""
        return cast(SampleType, Sample(a.s, b))

    def __init__(self, grammar: Union[IO[str], str, GrammaGrammar]):
        """
        grammar is either a GrammaGrammar object, a string containing GLF, or a file handle to a GLF file.
        """
        OperatorsImplementationSamplerMixin.__init__(self)

        self.grammar = GrammaGrammar.of(grammar)
        self.gfuncmap = {}
        self.coro_gfuncmap = {}
        self.gdfuncmap = {}
        self.coro_gdfuncmap = {}
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
                wd = gdfuncs.get('coro_' + ge.fname)
                if wd is not None:
                    self.coro_gdfuncmap[ge] = wd

            elif isinstance(ge, GFuncRef):
                w = gfuncs.get(ge.fname)
                if w is None:
                    missing.append(ge)
                else:
                    self.gfuncmap[ge] = w
                w = gfuncs.get('coro_' + ge.fname)
                if w is not None:
                    self.coro_gfuncmap[ge] = w

            elif isinstance(ge, GCode):
                self.gcodemap[ge] = GCodeWrap(ge)
            for gc in ge.get_code():
                self.gcodemap[gc] = GCodeWrap(gc)

        if len(missing) > 0:
            msg = f'missing handlers in {self.__class__.__name__}:  {",".join(ge.fname for ge in missing)}'
            log.error(msg)
            raise GrammaSamplerError(msg)

    def _(self, *args, return_=None, **kw):
        """
        a convenience for assigning to the sampler from gcode
            e.g.
                `_(x=5)` returns None
                `_(x, x=5)` returns the current value of sampler.x and updates x
        """

        ret = return_
        if len(args) > 0:
            ret = args[-1]

        for k, v in kw.items():
            setattr(self, k, v)

        return ret

    def exec(self, gc: GCode) -> Any:
        gcw = self.gcodemap[gc]
        return eval(gcw.compiled, gcode_globals, dict(self.__dict__, _=getattr(self, '_'), random=self.random))

    def eval_int(self, ge: Union[GTok, GCode]) -> int:
        return ge.as_int() if isinstance(ge, GTok) else cast(int, self.exec(ge))

    def eval_num(self, ge: Union[GTok, GCode]) -> Union[float, int]:
        return ge.as_num() if isinstance(ge, GTok) else cast(Union[float, int], self.exec(ge))

    def sample_start(self) -> SampleType:
        ge = self.grammar.ruledefs['start'].rhs
        return self.sample(ge)

    def coro_sample_start(self) -> SampleType:
        ge = self.grammar.ruledefs['start'].rhs
        return self.coro_sample(ge)

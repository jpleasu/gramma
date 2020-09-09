#!/usr/bin/env python3
"""
this script rewrites sampler_* and evaluate_denotation_* methods to corresponding coro_* methods.

coro_sample_GFunc (coro_evaluate_denototation_GDFunc) needs manual update to check for coro_* gfuncs(gdfuncs) and
issue 'yield from'.
"""

import ast
import inspect
import re
from ast import NodeVisitor, iter_fields, AST
from copy import deepcopy
from typing import Any, List, Union, Optional, Tuple, TypeVar, Generic, Dict

import astor

import gramma.samplers

_pat = re.compile('^(sample|evaluate_denotation)_')


def is_name_to_change(s: str):
    return _pat.search(s) is not None


class NamePathException(Exception):
    pass


class NamePath:
    __slots__ = 'path', 's'
    path: List[AST]
    s: str

    def __init__(self, n: AST):
        p = [n]
        while isinstance(p[0], (ast.Attribute, ast.Subscript)):
            p.insert(0, p[0].value)
        self.path = p
        self.s = self._compute_s()

    @staticmethod
    def _tostr(x: Union[AST, str]) -> str:
        if isinstance(x, str):
            return x
        elif isinstance(x, ast.Name):
            return x.id
        elif isinstance(x, ast.arg):
            return x.arg
        elif isinstance(x, ast.Attribute):
            return x.attr
        elif isinstance(x, ast.Subscript):
            return '[]'
        elif isinstance(x, ast.Str):
            return repr(x.s)
        else:
            raise NamePathException('not part of a path: %s' % type(x))

    def _compute_s(self) -> str:
        s = ''
        for x in self.path:
            if not isinstance(x, ast.Subscript) or s == '':
                s += '.'
            s += self._tostr(x)
        return s[1:]

    def __getitem__(self, slc):
        if isinstance(slc, int):
            return NamePath([self.path[slc]])
        elif isinstance(slc, slice):
            return NamePath(self.path[slc])
        return None

    def __len__(self):
        return len(self.path)

    def __hash__(self):
        return hash(self.s)

    def __eq__(self, other):
        if isinstance(other, NamePath):
            return self.s == other.s
        return self.s == str(other)

    def __repr__(self):
        return self.s


def walk(node: AST):
    yield node
    for field, value in iter_fields(node):
        if isinstance(value, list):
            for item in value:
                if isinstance(item, AST):
                    yield from walk(item)
        elif isinstance(value, AST):
            yield from walk(value)


T = TypeVar('T')


class Push(Generic[T]):
    l: List[T]

    def __init__(self, l: List[T], to_push: T):
        self.l = l
        self.to_push = to_push

    def __enter__(self):
        self.l.append(self.to_push)
        return self.l

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.l.pop()


class RewriteOp:
    op: str
    args: List[Any]

    def __init__(self, op, *args):
        self.op = op
        self.args = args

    def commit(self):
        if self.op == 'setfield':
            setattr(self.args[0], self.args[1], self.args[2])
        elif self.op == 'setfieldarray':
            getattr(self.args[0], self.args[1])[self.args[2]] = self.args[3]
        elif self.op == 'setfieldarrayslice':
            getattr(self.args[0], self.args[1])[self.args[2]:self.args[2] + 1] = self.args[3]


class RewritingVisitor(NodeVisitor):
    stack: List[AST]
    fieldi_stack: List[Tuple[str, Optional[int]]]
    added: int
    ops: Dict[ast.AST, List[RewriteOp]]

    def __init__(self):
        super().__init__()
        self.stack = []
        self.fieldi = []
        self.ops = {}

    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        with Push(self.stack, node):
            for field, value in iter_fields(node):
                if isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, AST):
                            with Push(self.fieldi, (field, i)):
                                self.visit(item)
                elif isinstance(value, AST):
                    with Push(self.fieldi, (field, None)):
                        self.visit(value)
        ops = self.ops.pop(node, None)
        if ops is not None:
            for op in reversed(ops):
                op.commit()

    @property
    def parent(self) -> Optional[AST]:
        if len(self.stack) > 0:
            return self.stack[-1]
        return None

    @property
    def field(self) -> str:
        return self.fieldi[-1][0]

    @property
    def i(self) -> Optional[int]:
        return self.fieldi[-1][1]

    def remove(self) -> None:
        self.replace([])

    def replace(self, elt: Union[ast.AST, List[ast.AST]]) -> None:
        if isinstance(elt, ast.AST):
            if self.i is None:
                self.ops.setdefault(self.parent, []).append(RewriteOp('setfield', self.parent, self.field, elt))
            else:
                self.ops.setdefault(self.parent, []).append(
                    RewriteOp('setfieldarray', self.parent, self.field, self.i, elt))
        elif isinstance(elt, list):
            if self.i is None:
                raise ValueError(f'can only replace parts of lists')
            self.ops.setdefault(self.parent, []).append(
                RewriteOp('setfieldarrayslice', self.parent, self.field, self.i, elt)
            )
        else:
            raise ValueError(f"can't replace with {elt}")


def meta(n: AST):
    """
    extract the metadata from a node for use in creating other nodes
    """
    return dict(lineno=n.lineno or 0, col_offset=n.col_offset or 0, end_lineno=n.end_lineno or 0,
                end_col_offset=n.end_col_offset or 0)


class ToCoro(RewritingVisitor):
    """
    convert a sampler method to coroutine style
    """

    def visit_Name(self, node: ast.Name) -> Any:
        if is_name_to_change(node.id):
            node.id = 'coro_' + node.id

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        if is_name_to_change(node.name):
            if node.name.startswith('sample'):
                node.returns = ast.parse('Generator[Union[GExpr,Sample],Sample,None]').body[0].value
            elif node.name.startswith('evaluate_denotation'):
                node.returns = ast.parse('Generator[Union[GExpr,Any],Any,None]').body[0].value
            node.name = 'coro_' + node.name
            self.generic_visit(node)
            kw = meta(node)

    def visit_Return(self, node: ast.Return) -> Any:
        self.generic_visit(node)

        kw = meta(node)
        y = ast.Yield(value=node.value, **kw)
        e = ast.Expr(value=y, **kw)
        self.replace(e)

    def visit_Call(self, node: ast.Call) -> Any:
        self.generic_visit(node)

        n = str(NamePath(node.func))
        if n == 'self.sample':
            kw = meta(node)
            self.replace(ast.Yield(value=node.args[0], **kw))
        elif n == 'self.evaluate_denotation':
            node.func.attr = 'coro_evaluate_denotation'


class FixYieldInComprehension(RewritingVisitor):
    def visit_Assign(self, node: ast.Assign) -> Any:
        if isinstance(node.value, ast.ListComp) and any(isinstance(c, ast.Yield) for c in walk(node.value.elt)):
            kw = meta(node)

            lcomp = node.value
            # assert the ListComp is over only one generator and its target is a Name
            iter_var = lcomp.generators[0].target
            source_iter = lcomp.generators[0].iter

            # assert there's only one target of the Assign, and it's a Name
            target_list = node.targets[0]
            target_list_load = deepcopy(target_list)
            target_list_load.ctx = ast.Load()
            emptylist = ast.List(elts=[], ctx=ast.Load(), **kw)

            append_stmt = ast.Expr(value=ast.Call(
                func=ast.Attribute(value=target_list_load, attr='append', ctx=ast.Load(), **kw),
                args=[lcomp.elt],
                keywords=[],
                **kw), **kw)

            forloop = ast.For(
                target=iter_var,
                iter=source_iter,
                orelse=[],
                body=[
                    append_stmt
                ], **kw)

            # replacing with an If, just to have something to put in a single slot
            replacements = [
                ast.Assign(targets=[target_list], value=emptylist, **kw),
                forloop
            ]
            replacement = ast.If(test=ast.Constant(value=True, **kw),
                                 body=replacements,
                                 orelse=[],
                                 **kw)
            self.replace(replacements)


src = inspect.getsource(gramma.samplers.interpreter.OperatorsImplementationSamplerMixin)
m = ast.parse(src)
# mm = deepcopy(m)

m.body[0].name = 'Coro' + m.body[0].name
# import astpretty
# astpretty.pprint(m)

# remove all but the methods we care to change (and assignments)
l = []
for i, c in enumerate(m.body[0].body):
    if isinstance(c, ast.Assign):
        if is_name_to_change(c.targets[0].id):
            continue
    elif isinstance(c, ast.FunctionDef):
        if is_name_to_change(c.name):
            continue
    l.append(i)

for i in reversed(l):
    del m.body[0].body[i]

# change names, replace self.sample with yield and return with yield
ToCoro().visit(m)

FixYieldInComprehension().visit(m)

# verify we can at least compile
code = compile(m, 'generated.py', 'exec')
exec(code, vars(gramma.samplers.interpreter))
AsyncOperatorsImplementationSamplerMixin = vars(gramma.samplers.interpreter)['CoroOperatorsImplementationSamplerMixin']
from astor.string_repr import pretty_string as astor_pretty_string
from astor.source_repr import split_lines as astor_split_lines

MAXLINE = 120


def pretty_string(*args, **kwargs):
    kwargs['max_line'] = MAXLINE
    return astor_pretty_string(*args, **kwargs)


def pretty_source(source):
    return ''.join(astor_split_lines(source, maxline=MAXLINE))


print(astor.to_source(m, pretty_string=pretty_string, pretty_source=pretty_source))

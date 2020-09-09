#!/usr/bin/env python3
from typing import Set, Dict, TypeVar, Generic, Iterable, Optional, List, Any, Protocol, AbstractSet


class PushesAndPops(Protocol):  # pragma: no cover
    def push(self, arg: Any) -> None:
        ...

    def pop(self) -> Any:
        ...


ObjT = TypeVar('ObjT', bound=PushesAndPops)
ArgT = TypeVar('ArgT')


# I don't think this can be typed correctly atm, see https://github.com/python/mypy/issues/3151
class Context(Generic[ObjT, ArgT]):
    __slots__ = 'obj', 'arg'
    obj: ObjT
    arg: Optional[ArgT]

    def __init__(self, obj: ObjT, arg: Optional[ArgT]):
        self.obj = obj
        self.arg = arg

    def __enter__(self) -> ObjT:
        self.obj.push(self.arg)
        return self.obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.obj.pop()


T = TypeVar('T')


class SetStack(Generic[T]):
    """
        A stack of sets, if the top doesn't have a key, check the next, ...
    """
    __slots__ = 'stack',

    stack: List[Set[T]]

    def __init__(self):
        self.stack = [set()]

    @property
    def top(self) -> Set[T]:
        return self.stack[-1]

    def add(self, item: T) -> None:
        self.top.add(item)

    def update(self, items: Iterable[T]) -> None:
        self.top.update(items)

    def __contains__(self, item: T) -> bool:
        for s in reversed(self.stack):
            if item in s:
                return True
        return False

    def push(self, s: Optional[Iterable[T]] = None) -> None:
        self.stack.append(set())
        if s is not None:
            self.update(s)

    def pop(self) -> Set[T]:
        return self.stack.pop()

    def context(self, s: Optional[Iterable[T]] = None) -> Context['SetStack[T]', Iterable[T]]:
        return Context(self, s)


K = TypeVar('K')
V = TypeVar('V')


class DictStack(Generic[K, V]):
    """
        A stack of dictionaries, if the top doesn't have a key, check the next, ...

        Assumes that None is not a valid value.
    """
    __slots__ = 'stack',

    stack: List[Dict[K, V]]

    def __init__(self):
        self.stack = [{}]

    @property
    def top(self) -> Dict[K, V]:
        return self.stack[-1]

    def __contains__(self, key):
        for d in reversed(self.stack):
            if key in d:
                return True
        return False

    def get(self, key: K) -> Optional[V]:
        for d in reversed(self.stack):
            item = d.get(key)
            if item is not None:
                return item
        return None

    __getitem__ = get

    def __setitem__(self, key: K, item: V) -> None:
        self.top[key] = item

    def update(self, items: Dict[K, V]) -> None:
        self.top.update(items)

    def push(self, d: Optional[Dict[K, V]] = None) -> None:
        self.stack.append({})
        if d is not None:
            self.update(d)

    def pop(self) -> Dict[K, V]:
        return self.stack.pop()

    def context(self, d: Optional[Dict[K, V]] = None) -> Context['DictStack[K, V]', Dict[K, V]]:
        return Context(self, d)

    def __str__(self):
        s = ','.join(str(d) for d in self.stack if len(d) > 0)
        if s == '':
            return '{}'
        return s

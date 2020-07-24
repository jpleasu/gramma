#!/usr/bin/env python3
from typing import Set, Union, Dict, TypeVar, Generic, Iterable, Optional

T = TypeVar('T')


class SetStack(Generic[T]):
    """
        A stack of sets, if the top doesn't have a key, check the next, ...
    """
    __slots__ = 'local', 'parent'

    parent: Union[Set[T], 'SetStack[T]']
    local: Set[T]

    def __init__(self, parent=None):
        self.parent = parent if parent is not None else set()
        self.local = set()

    def add(self, item: T):
        self.local.add(item)

    def update(self, items: Iterable[T]):
        self.local.update(items)

    def __contains__(self, item) -> bool:
        if item in self.local:
            return True
        return item in self.parent


K = TypeVar('K')
V = TypeVar('V')


class DictStack(Generic[K, V]):
    """
        A stack of dictionaries, if the top doesn't have a key, check the next, ...
    """
    __slots__ = 'local', 'parent'

    parent: Union[Dict[K, V], 'DictStack[K, V]']
    local: Dict[K, V]

    def __init__(self, parent=None):
        self.parent = parent if parent is not None else dict()
        self.local = {}

    def get(self, key: K) -> Optional[V]:
        item = self.local.get(key)
        if item is not None:
            return item
        return self.parent.get(key)

    __getitem__ = get

    def __setitem__(self, key: K, item: V):
        self.local[key] = item

    def update(self, items: Dict[K, V]):
        self.local.update(items)


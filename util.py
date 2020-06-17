#!/usr/bin/env python3

class SetStack(object):
    """
        A stack of sets, if the top doesn't have a key, check the next, ...
    """
    __slots__ = 'local', 'parent'

    def __init__(self, parent=set()):
        self.parent = parent
        self.local = set()

    def add(self, item):
        self.local.add(item)

    def update(self, items):
        self.local.update(items)

    def __contains__(self, item):
        if item in self.local:
            return True
        return item in self.parent


class DictStack(object):
    """
        A stack of dictionaries, if the top doesn't have a key, check the next, ...
    """
    __slots__ = 'local', 'parent'

    def __init__(self, parent={}):
        self.parent = parent
        self.local = {}

    def get(self, key):
        item = self.local.get(key)
        if item is not None:
            return item
        return self.parent.get(key)

    __getitem__ = get

    def __setitem__(self, key, item):
        self.local[key] = item

    def update(self, items):
        self.local.update(items)



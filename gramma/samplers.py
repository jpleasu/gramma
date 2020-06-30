#!/usr/bin/env python3

from .core import GrammaSampler
from .sideeffects import Tracer

class TracingSampler(GrammaSampler):
    __slots__ = 'tracer',

    def __init__(self, grammar=None, **params):
        GrammaSampler.__init__(self, grammar, **params)
        self.tracer = Tracer()
        self.add_sideeffects(self.tracer)

    def sample_tracetree(self, ge=None, randstate=None):
        if randstate is not None:
            self.random.set_state(randstate)
        self.sample(ge)
        return self.tracer.tracetree

# vim: ts=4 sw=4

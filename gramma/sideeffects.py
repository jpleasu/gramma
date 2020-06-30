#!/usr/bin/env python3

import copy
import sys

from .core import GAlt, GCat, GTern, GFunc, GRule, GRep, GRange, GTok
from .core import SideEffect, CacheConfig, log
from .util import defaultdict


class GeneralDepthTracker(SideEffect):
    __slots__ = 'pred', 'varname', 'initial_value'

    def __init__(self, pred=None, varname='depth', initial_value=0):
        if pred is None:
            pred = lambda ge: True
        self.pred = pred
        self.varname = varname
        self.initial_value = initial_value

    def get_reset_states(self):
        return set([pysa.NamePath(self.varname)])

    def reset_state(self, state):
        setattr(state, self.varname, self.initial_value)

    def push(self, x, ge):
        if self.pred(ge):
            setattr(x.state, self.varname, getattr(x.state, self.varname) + 1)
            return True
        return False

    def pop(self, x, w, s):
        if w:
            setattr(x.state, self.varname, getattr(x.state, self.varname) - 1)


class DepthTracker(SideEffect):
    def reset_state(self, state):
        state.depth = 0

    def push(self, x, ge):
        x.state.depth += 1
        return True

    def pop(self, x, w, s):
        x.state.depth -= 1


class RuleDepthTracker(SideEffect):
    def reset_state(self, state):
        state.depth = 0

    def push(self, x, ge):
        if isinstance(ge, GRule):
            x.state.depth += 1
            return True
        return False

    def pop(self, x, w, s):
        if w:
            x.state.depth -= 1


class Tracer(SideEffect):
    __slots__ = 'tt', 'tracetree'

    def reset_state(self, state):
        self.tracetree = None

    def push(self, x, ge):
        if self.tracetree is None:
            self.tt = self.tracetree = TraceNode(ge)
        else:
            self.tt = self.tt.add_child(ge)
        m = ge.get_meta()
        if m.uses_random:
            self.tt.inrand = x.random.r.get_state()
        if m.statevar_uses:
            self.tt.instate = type('InState', (), {})()
            for varname in m.statevar_uses:
                setattr(self.tt.instate, varname, copy.deepcopy(getattr(x.state, varname)))
        return None

    def pop(self, x, w, s):
        m = self.tt.ge.get_meta()

        self.tt.s = s
        if m.uses_random:
            self.tt.outrand = x.random.r.get_state()
        if m.statevar_defs:
            self.tt.outstate = type('OutState', (), {})()
            for varname in m.statevar_defs:
                setattr(self.tt.outstate, varname, copy.deepcopy(getattr(x.state, varname)))

        self.tt = self.tt.parent


class TraceNode(object):
    """
        a node of the tracetree

        Tracer populates each TraceNode w/ incoming and outgoing rand and state
        values.
        - __when__
            - __it's_saved_to__

        - if a node samples
            - n.children
        - if a node uses_random,
            - on enter
                - n.inrand
            - before sampling child
                - child.inrand
            - after sampling child
                - child.outrand
            - on return
                - n.outrand
        - if a node uses state:
            - on enter
                - n.instate
            - after sampling child
                - child.outstate
        - if a node defs state:
            - before sampling child
                - child.instate
            - on return
                - n.outstate

        When resampling, we can "replay" different combinations of the above
        inputs to a node.

        - rand and other state can be set beforehand
            - random state can be set
                load_rand('r').load('var','var0').e
        - (sub)samples
            - for all but gfuncs,

    """
    __slots__ = 'ge', 'parent', 'children', 's', 'inrand', 'outrand', 'instate', 'outstate'

    def __init__(self, ge):
        self.ge = ge
        self.parent = None
        self.children = []
        self.s = None
        self.inrand = None
        self.outrand = None
        self.instate = None
        self.outstate = None

    def add_child(self, ge):
        c = TraceNode(ge)
        c.parent = self
        self.children.append(c)
        return c

    def dump(self, indent=0, out=sys.stdout):
        print('%s%s -> "%s"' % ('  ' * indent, self.ge, self.s), file=out)
        for c in self.children:
            c.dump(indent + 1, out)

    def inbounds(self, i, j):
        """[i,j) contained in [0,len(self.s))?"""
        return 0 <= i and j <= len(self.s)

    def child_containing(self, i, j=None, d=0):
        if j is None or j < i + 1:
            j = i + 1

        # print('%s[%d,%d) %s' % (' '*d, i,j,self.ge))

        if isinstance(self.ge, (GRule, GTern, GAlt)):
            return self.children[0].child_containing(i, j, d + 1)

        # don't descend into GFuncs
        if isinstance(self.ge, (GCat, GRep)):
            # i         v
            #   aaaa  bbbb   cccc
            #
            o = 0
            for c in self.children:
                x = c.child_containing(i - o, j - o, d + 1)
                if x is not None:
                    return x
                o += len(c.s)
            if self.inbounds(i, j):
                return self
            return None

        if isinstance(self.ge, (GTok, GFunc, GRange)):
            return self if self.inbounds(i, j) else None

        raise GrammaParseError('unknown expression (%s)%s' % (type(self.ge), self.ge))

    def first(self, pred):
        for n in self.gennodes():
            if pred(n):
                return n
        return None

    def first_rule(self, rname):
        return self.first(lambda n: n.ge.is_rule(rname))

    def last(self, pred):
        for n in self.gennodesr():
            if pred(n):
                return n
        return None

    def gennodes(self):
        yield self
        for c in self.children:
            for cc in c.gennodes():
                yield cc

    def gennodesr(self):
        """reverse node generator"""
        for c in reversed(self.children):
            for cc in c.gennodesr():
                yield cc
        yield self

    def depth(self, pred=lambda x: True):
        """
            # of ancestors that satisfy pred
        """
        n = self.parent
        d = 0
        while n is not None:
            if pred(n):
                d += 1
            n = n.parent
        return d

    def resample_mostly(self, grammar, pred, factor=10):
        """
            like resample, but non-resampled nodes aren't definitized, they're
            just biased toward their previous decision.

            factor is how much more likely the previous selection should be at
            each alternation and ternary.

            Even with a factor of 0, this resample is useful for preserving the
            tracetree structure for future resample operations.
        """
        cachecfg = CacheConfig()

        def recurse(t):
            """
                return resampled, gexpr
                    where resampled is True if a subexpression will be resampled
            """
            if pred(t):
                outrand = t.last(lambda n: n.ge.get_meta().uses_random)
                if outrand is not None:
                    return True, t.ge.copy()

            ## mostly definitize ##
            ge = t.ge
            if isinstance(ge, (GAlt, GTern)):
                tc = t.children[0]
                b, tcc = recurse(tc)
                cmap = defaultdict(lambda c: (1, c))
                cmap[tc.ge] = (factor, tcc)

                weights, children = zip(*[cmap[c] for c in ge.children])
                return b, GAlt(weights, children)
            elif isinstance(ge, GRule):
                return recurse(t.children[0])
            elif isinstance(ge, (GCat, GRep)):
                l = [recurse(c) for c in t.children]
                return any(r for (r, ge) in l), GCat([ge for (r, ge) in l])
            elif isinstance(ge, GRange):
                return False, GTok.from_str(t.s)
            elif isinstance(ge, GTok):
                return False, ge.copy()
            elif isinstance(ge, GFunc):
                l = [recurse(c) for c in t.children]
                if not any(r for (r, ge) in l):
                    # preserve the function call
                    return False, GFunc(ge.fname, [arg[1] for arg in l], ge.gf)
                fargmap = {}
                for i, c in enumerate(t.children):
                    fargmap.setdefault(c.ge, []).append(i)
                args = []
                for a in t.ge.fargs:
                    ta = fargmap.get(a, [])
                    if len(ta) > 1:
                        log.warning(
                            'argument sampled multiple times in %s(..,%s,..): %s, resampling original expression' % (
                                ge.fname, a, ta))
                        # log.warning(str(fargmap))
                        args.append(a.copy())
                    elif len(ta) == 0:
                        # this argument wasn't sampled.. use a copy of the
                        # original
                        args.append(a.copy())
                    else:
                        # use the computed recursion on the tracenode child
                        args.append(l[ta[0]][1])
                return True, GFunc(ge.fname, args, ge.gf)
            else:
                raise ValueError('unknown GExpr node type: %s' % type(ge))

        b, ge = recurse(self)
        return ge.simplify(), cachecfg

    def resample(self, grammar, pred):
        """
            computes ge, a GExpr that resamples only the nodes satisfying pred.

            computes cfg, a CacheConfig populated with any extra random and
            state values needed by ge.

            return ge,cfg
        """

        cachecfg = CacheConfig()

        # the enumeration below is done left to right, so our accumulation of
        # statevar should be correct
        # these are the statevars defed by defintinitized elements.. e.g. that
        # won't be available unless they're explicitly provided
        modified_statevars = set()

        def recurse(t):
            """
                return resampled, gexpr
                    where resampled is True if a subexpression will be resampled
            """
            meta = t.ge.get_meta()

            if pred(t):
                if modified_statevars.isdisjoint(meta.statevar_uses):
                    for v in meta.statevar_defs:
                        modified_statevars.discard(v)
                    return True, t.ge.copy()
                l = []
                for varname in modified_statevars & meta.statevar_uses:
                    slot = cachecfg.new_state(getattr(t.instate, varname))
                    l.append(grammar.parse('''load('%s','%s')''' % (varname, slot)))

                l.append(t.ge.copy())
                for v in meta.statevar_defs:
                    modified_statevars.discard(v)

                return True, GCat(l)

                ## generate new random ##
                # slot=cachecfg.new_randstate(outrand.outrand)
                # return GCat([grammar.parse('reseed_rand()'),t.ge.copy(),grammar.parse("load_rand('"+slot+"')")])
                # return True, GCat([GTok.from_str('>>>>>>>'),t.ge.copy(),GTok.from_str('<<<<<<<<')])

            modified_statevars0 = modified_statevars.copy()
            for v in meta.statevar_defs:
                modified_statevars.add(v)

            ## definitize ##
            ge = t.ge
            if isinstance(ge, GTern):
                return recurse(t.children[0])
            elif isinstance(ge, GAlt):
                return recurse(t.children[0])
            elif isinstance(ge, GRule):
                return recurse(t.children[0])
            elif isinstance(ge, (GCat, GRep)):
                l = [recurse(c) for c in t.children]
                return any(r for (r, ge) in l), GCat([ge for (r, ge) in l])
            elif isinstance(ge, GRange):
                return False, GTok.from_str(t.s)
            elif isinstance(ge, GTok):
                return False, ge.copy()
            elif isinstance(ge, GFunc):
                l = [recurse(c) for c in t.children]
                if not any(r for (r, ge) in l):
                    return False, GTok.from_str(t.s)
                fargmap = {}
                for i, c in enumerate(t.children):
                    fargmap.setdefault(c.ge, []).append(i)
                args = []
                for a in t.ge.fargs:
                    ta = fargmap.get(a, [])
                    if len(ta) > 1:
                        log.warning(
                            'argument sampled multiple times in %s(..,%s,..): %s, resampling original expression' % (
                                ge.fname, a, ta))
                        # log.warning(str(fargmap))
                        args.append(a.copy())
                    elif len(ta) == 0:
                        # this argument wasn't sampled.. use a copy of the
                        # original
                        args.append(a.copy())
                    else:
                        # use the computed recursion on the tracenode child
                        args.append(l[ta[0]][1])

                newgf = GFunc(ge.fname, args, ge.gf)

                # if this function needs state, load what it had last time
                # XXX it's getting a new random, so this isn't definitized!
                # given that the gfunc might use random in and amongst samples,
                # AND we might use rand in combination with samples, it's messy
                # to recreate..
                #   load_rand('gf_inrand').gf(arg1.load_rand('arg1_outrand'), ...).reseed_rand()
                if modified_statevars0.isdisjoint(meta.statevar_uses):
                    return True, newgf

                l = []
                for varname in modified_statevars0 & meta.statevar_uses:
                    slot = cachecfg.new_state(getattr(t.instate, varname))
                    l.append(grammar.parse('''load('%s','%s')''' % (varname, slot)))
                l.append(newgf)

                return True, GCat(l)
            else:
                raise ValueError('unknown GExpr node type: %s' % type(ge))

        b, ge = recurse(self)
        return ge.simplify(), cachecfg

# vim: ts=4 sw=4

import textwrap
from functools import cached_property
from typing import Tuple, Union, List, Set, cast, Optional, Dict, Any

import colorama
import networkx as nx
from termcolor import colored

from . import GrammaGrammar
from .parser import GExpr, GRuleRef, GAlt, GInternal, GCode

OpType = Union[
    Tuple[str, int, str],
    Tuple[str, int, int],
    Tuple[str, int, int, str],
]


class Rewriter:
    src: str
    ops: List[OpType]

    def __init__(self, src: str):
        self.src = src
        self.ops = []

    def insert(self, pos: int, s: str) -> None:
        self.ops.append(('i', pos, s))

    def delete(self, pos0: int, pos1: int) -> None:
        self.ops.append(('d', pos0, pos1))

    def replace(self, pos0: int, pos1: int, s: str) -> None:
        self.ops.append(('r', pos0, pos1, s))

    def doit(self) -> str:
        src = self.src
        parts = []
        for op in reversed(self.ops):
            c = op[0]
            if c == 'i':
                pos = op[1]
                parts.append(src[pos:])
                parts.append(cast(str, op[2]))
                src = src[:pos]
            elif c == 'd':
                pos0, pos1 = cast(Tuple[int, int], op[1:])
                parts.append(src[pos1:])
                src = src[:pos0]
            elif c == 'r':
                pos0, pos1, s = cast(Tuple[int, int, str], op[1:])
                parts.append(src[pos1:])
                parts.append(s)
                src = src[:pos0]
        parts.append(src)
        return ''.join(reversed(parts))


class DFSNode:
    def __init__(self, parent: GExpr, gel: List[GExpr], i: int = 0):
        self.parent = parent
        self.gel = gel
        self.i = i

    @property
    def ge(self) -> GExpr:
        return self.gel[self.i]

    def next(self) -> bool:
        self.i += 1
        return self.i < len(self.gel)

    def copy(self):
        return DFSNode(self.parent, self.gel, self.i)

    def __repr__(self):
        return f'{self.parent}[{self.i}] = {self.ge}'


Edge = Tuple[int, int]
Cycle = Tuple[int, ...]


# noinspection PyShadowingNames
class EdgeInfo:
    ge: GExpr
    i: int
    recursion: bool
    meets_cycles: Set[Cycle]

    def __init__(self, ge: GExpr, i: int, recursion: bool = False):
        self.ge = ge
        self.i = i
        self.recursion = recursion
        self.meets_cycles = set()

    def meets(self, cycle: Cycle):
        self.meets_cycles.add(cycle)

    @property
    def ncycles(self):
        return len(self.meets_cycles)


class CFG:
    edge_info: Dict[Edge, EdgeInfo]
    edges: List[Edge]
    dg: nx.DiGraph
    g2n: Dict[GExpr, int]
    n2g: Dict[int, GExpr]
    cycles: Set[Cycle]

    def __init__(self, g: GrammaGrammar):
        self.g2n = {}
        self.edges = []
        self.edge_info = {}

        for ge in g.walk():
            n0 = self.node(ge)
            off = 0
            if isinstance(ge, GRuleRef):
                e = (n0, self.node(g.ruledefs[ge.rname].rhs))
                self.edges.append(e)
                self.edge_info[e] = EdgeInfo(ge, 0, recursion=True)
                off += 1

            if isinstance(ge, GInternal):
                for i, n1 in enumerate(ge.children):
                    e = (n0, self.node(n1))
                    self.edges.append(e)
                    self.edge_info[e] = EdgeInfo(ge, i + off)

        self.n2g = dict((v, k) for k, v in self.g2n.items())

        self.dg = nx.DiGraph(self.edges)

        self.cycles = set(tuple(lc) for lc in nx.simple_cycles(self.dg))
        self.cyclenum = {}
        for c in self.cycles:
            self.cyclenum[c] = 1 + len(self.cyclenum)

        for c in self.cycles:
            for edge in zip(c, c[1:] + c[:1]):
                self.edge_info[cast(Edge, edge)].meets(c)

    def node(self, ge: GExpr) -> int:
        return self.g2n.setdefault(ge, len(self.g2n))


class GLFColorize(Rewriter):
    def __init__(self, glf: str, parent: GExpr, whole_lines: bool = False):
        self.parent = parent
        if whole_lines:
            sloc = parent.location.whole_lines(glf)
        else:
            sloc = parent.location
        s = sloc(glf)
        self.o0 = sloc.start_pos
        super().__init__(s)

    def __call__(self, child: GExpr, *colors: str, **colorskw: Any):
        default = colorskw.pop('default', None)
        default_start = colorskw.pop('default_start', None)
        default_end = colorskw.pop('default_end', default_start)

        if child.location is None:
            repl = default
            start, end = default_start, default_end
        else:
            start, end = child.location.start_pos - self.o0, child.location.end_pos - self.o0
            repl = self.src[start:end]
        repl = '\n'.join(colored(part, *colors, **colorskw) for part in repl.split('\n'))
        self.replace(start, end, repl)

    def print_numbered_lnes(self, ln: int):
        for line in self.doit().split('\n'):
            print(f'{ln:8} : {line}')
            ln += 1


class GLFAnalyzer:
    def __init__(self, glf: str):
        self.glf = glf

    def compute_feedback_arcs(self) -> List[List[DFSNode]]:
        grammar = self.grammar

        stack: List[DFSNode] = [DFSNode(GRuleRef('start', []), [grammar.ruledefs['start'].rhs])]
        feedbackarcs: List[List[DFSNode]] = []
        visited: Set[GExpr] = set()

        while True:
            n = stack[-1]
            ge = n.ge

            stk0 = [nn.ge for nn in stack[:-1]]
            j: Optional[int] = stk0.index(ge) if ge in stk0 else None
            if j is not None:
                feedbackarcs.append([nn.copy() for nn in stack[j + 1:]])
            elif ge not in visited:
                visited.add(ge)
                if isinstance(ge, GRuleRef):
                    stack.append(DFSNode(ge, [grammar.ruledefs[ge.rname].rhs] + ge.rargs))
                    continue
                elif isinstance(ge, GInternal) and len(ge.children) > 0:
                    stack.append(DFSNode(ge, ge.children))
                    continue

            while len(stack) > 0:
                n = stack[-1]
                if n.next():
                    break
                stack.pop()

            if len(stack) == 0:
                break
        return feedbackarcs

    def print_alt_edgeinfo(self, ei: EdgeInfo) -> None:
        colorize = GLFColorize(self.glf, ei.ge, whole_lines=True)
        weight = ei.ge.weights[ei.i]
        child = ei.ge.children[ei.i]
        colorize(weight, 'red', attrs=['bold'], default=' X ', default_start=child.location.start_pos)
        colorize(child, 'green', attrs=['bold'])
        colorize.print_numbered_lnes(ln=ei.ge.location.line)

    def print_cycle(self, cycle: Cycle):
        cfg = self.cfg
        print(f'== cycle #{cfg.cyclenum[cycle]} ==')
        for n0, n1 in zip(cycle, cycle[1:] + cycle[:1]):
            e = (n0, n1)
            ei = cfg.edge_info[e]

            ge0 = cfg.n2g[n0]
            ge1 = cfg.n2g[n1]
            colorize = GLFColorize(self.glf, ge0, whole_lines=True)
            colorize(ge0, 'green', attrs=['bold'])
            # if not ei.recursion:
            #     colorize(ge1, 'green')
            colorize.print_numbered_lnes(ln=ge0.location.line)
            print('  --')

    @cached_property
    def grammar(self):
        return GrammaGrammar(self.glf)

    @cached_property
    def cfg(self):
        return CFG(self.grammar)

    def report_cycle_breaking_alternatives(self):
        glf = self.glf
        cfg = self.cfg

        print('=======================================')
        print('===== Cycle breaking alternatives =====')
        print('=======================================')
        # take a copy of cycles so we can modify it in the greedy set cover below
        cycles = set(cfg.cycles)

        fail = False
        alt_edges = set(e for e, ei in cfg.edge_info.items() if isinstance(ei.ge, GAlt))
        for c in cycles:
            meets_alt = False
            for edge in zip(c, c[1:] + c[:1]):
                if edge in alt_edges:
                    meets_alt = True
                    break
            else:
                if not meets_alt:
                    fail = True
                    print("== cycle NOT controlled with an alternation! ==")
                    self.print_cycle(c)
        if fail:
            return

        def priority(ei):
            """
            eliminate cycles that meet GCode alternatives first, then choose alternatives that
            cover a maximum of cycles for a greedy cover
            """
            return isinstance(ei.ge.weights[ei.i], GCode) and len(ei.meets_cycles) > 0, len(ei.meets_cycles)

        eis = [cfg.edge_info[e] for e in alt_edges]
        # remove cycles containing the higest priority alternative that remains until all cycles are covered
        othercycles: Dict[EdgeInfo, Set[Cycle]] = dict((ei, set()) for ei in eis)
        while len(cycles) > 0:
            ei = max(eis, key=priority)
            self.print_alt_edgeinfo(ei)

            p = 's' if len(ei.meets_cycles) > 1 else ''
            print(
                f'        {ei.ncycles} cycle{p} broken: {",".join("#" + str(cfg.cyclenum[cyc]) for cyc in ei.meets_cycles)}')
            o = othercycles[ei]
            if len(o) > 0:
                p = 's' if len(othercycles) > 1 else ''
                print(f'          {len(o)} other cycle{p} met: {",".join("#" + str(cfg.cyclenum[cyc]) for cyc in o)}')
            s = set(ei.meets_cycles)
            cycles -= s
            for ei in eis:
                othercycles[ei] |= (s & ei.meets_cycles)
                ei.meets_cycles &= cycles
            print('  --')
        print()
        print()

    def backarcs(self):
        glf = self.glf

        rw = GLFAnalyzer(glf)
        for fal in sorted(rw.compute_feedback_arcs(),
                          key=lambda fal: len([n for n in fal if isinstance(n.parent, GAlt) and n.parent.dynamic])):
            print('========')
            for n in fal[:-1]:
                if isinstance(n.parent, GAlt):
                    print(f'{str(n.parent.weights[n.i]):20}  {n.parent} -> {n.i}')
                else:
                    print(f'                      {n.parent} -> {n.i}')
            n = fal[-1]
            if isinstance(n.parent, GAlt):
                print(f'*{str(n.parent.weights[n.i]):19}  {n.parent} -> {n.i}    -^  {n.ge}')
            else:
                print(f'*                     {n.parent} -> {n.i}    -^ {n.ge}')
        print()
        print()

    def dump_cycles(self):
        print('=======================================')
        print('=====          All cycles        ======')
        print('=======================================')
        for cycle in self.cfg.cycles:
            self.print_cycle(cycle)
        print()
        print()


def main(main_args: Optional[List[str]] = None) -> None:
    import argparse
    from argparse import RawTextHelpFormatter

    parser = argparse.ArgumentParser(description='Analyze a gramma GLF file',
                                     formatter_class=RawTextHelpFormatter)

    parser.add_argument('glf', metavar='GLF_IN', type=argparse.FileType(),
                        help='input GLF file')
    parser.add_argument('--color', type=str, choices=['always', 'auto', 'never'], default='auto',
                        help='input GLF file')
    parser.add_argument('-dc', '--dump-cycles', dest='dump_cycles', action='store_true', default=False,
                        help='show all cycles due to recursion')
    parser.add_argument('-cba', '--find-cycle-breaking-alternatives', dest='find_cycle_breaking_alts',
                        action='store_true',
                        default=False,
                        help='find alternation coefficients that can be used to break cycles due to recursion\n'
                             'by first eliminating cycles broken by dynamic weights, then proceeding through \n'
                             'alternatives greedly, that is, in decreasing number of cycles broken'
                        )

    parser.add_argument('-v', dest='verbosity', action='count', default=0,
                        help='verbosity level.  repeat for more (-vvvv) ')

    args = parser.parse_args(args=main_args)
    if args.color == 'auto':
        colorama.init()
    else:
        colorama.init(strip=args.color == 'never')
    a = GLFAnalyzer(args.glf.read())

    did_anything = False

    if args.dump_cycles:
        a.dump_cycles()
        did_anything = True

    if args.find_cycle_breaking_alts:
        a.report_cycle_breaking_alternatives()
        did_anything = True

    if not did_anything:
        parser.print_help()

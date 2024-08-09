from enum import Enum
from typing import List

from qat.purr.backends.graph import BasicBlock, ControlFlowGraph


class CodegenPassRegistry(Enum):
    CFG = "CFG"
    CTRL_HW = "CTRL_HW"
    ITER_BOUNDS = "ITER_BOUNDS"
    QUANTUM_TARGETS = "QUANTUM_TARGETS"
    SWEEP_DECOMPOSITION = "SWEEP_DECOMPOSITION"
    SCOPE_BALANCING = "SCOPE_BALANCING"
    TIMELINE = "TIMELINE"


class DfsTraversal:
    def __init__(self):
        self._entered: List[BasicBlock] = []

    def clear(self):
        self._entered.clear()

    def run(self, graph: ControlFlowGraph):
        self.clear()
        self._visit(graph.entry, graph)

    def _visit(self, node: BasicBlock, graph: ControlFlowGraph):
        self.enter(node)
        self._entered.append(node)
        for neighbour in graph.out_nbrs(node):
            if neighbour not in self._entered:
                self._visit(neighbour, graph)
        self.exit(node)

    def enter(self, node: BasicBlock):
        pass

    def exit(self, node: BasicBlock):
        pass

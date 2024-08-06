from typing import List

from qat.purr.compiler.instructions import Instruction


class BasicBlock:
    def __init__(self, indices=None):
        self.indices: List[int] = indices or []

    def head(self):
        if self.indices:
            return self.indices[0]
        return None

    def tail(self):
        if self.indices:
            return self.indices[-1]
        return None

    def is_empty(self):
        return len(self.indices) == 0

    def iterator(self):
        return iter(self.indices)


class Flow:
    def __init__(self, src=None, dest=None):
        self.src: BasicBlock = src
        self.dest: BasicBlock = dest


class ControlFlowGraph:
    def __init__(self, nodes=None, edges=None):
        self.nodes: List[BasicBlock] = nodes or []
        self.edges: List[Flow] = edges or []
        self.entry = None

    def get_or_create_node(self, header: int) -> BasicBlock:
        node = next((n for n in self.nodes if n.head() == header), None)
        if not node:
            node = BasicBlock([header])
            self.nodes.append(node)

        self.entry = self.entry or node
        return node

    def get_or_create_edge(self, src: BasicBlock, dest: BasicBlock) -> Flow:
        edge = next((f for f in self.edges if f.src == src and f.dest == dest), None)
        if not edge:
            edge = Flow(src, dest)
            self.edges.append(edge)
        return edge

    def out_nbrs(self, node) -> List[BasicBlock]:
        return [e.dest for e in self.edges if e.src == node]

    def in_nbrs(self, node) -> List[BasicBlock]:
        return [e.src for e in self.edges if e.dest == node]

    def out_edges(self, node) -> List[Flow]:
        return [e for e in self.edges if e.src == node]

    def in_edges(self, node) -> List[Flow]:
        return [e for e in self.edges if e.dest == node]


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


class EmitterMixin:
    def __init__(self, instructions: List[Instruction]):
        self.instructions = instructions

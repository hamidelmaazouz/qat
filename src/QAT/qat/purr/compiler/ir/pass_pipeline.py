from enum import Enum, auto
from typing import List

from qat.purr.compiler.ir.pass_base import PassInfoMixin, PassModel, PassResultSet


class PassManager(PassInfoMixin):
    """
    The pass manager acts as a pass itself. In doing so, it runs a sequence of passes over some unit of IR
    and aggregates results from them.

    Result aggregation could potentially involve caching invalidation concepts as described in
    PassResultSet.

    For today's needs, it just accumulates and returns a set of results from the passes.
    """

    def __init__(self):
        self.passes: List[PassModel] = []

    def run(self, ir, *args, **kwargs):
        global_rs = PassResultSet()
        for p in self.passes:
            pass_rs = p.run(ir, *args, **kwargs)
            global_rs.update(pass_rs)
        return global_rs

    def add(self, pass_obj):
        self.passes.append(PassModel(pass_obj))


class PassRegistry(Enum):
    CFG = "CFG"
    CTRL_HW = "CTRL_HW"
    ITER_BOUNDS = "ITER_BOUNDS"
    QUANTUM_TARGETS = "QUANTUM_TARGETS"
    SWEEP_DECOMPOSITION = "SWEEP_DECOMPOSITION"
    SCOPE_BALANCING = "SCOPE_BALANCING"
    TIMELINE = "TIMELINE"

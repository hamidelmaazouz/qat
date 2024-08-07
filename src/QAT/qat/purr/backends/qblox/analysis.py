import numpy as np

from qat.purr.backends.concept import PassInfoMixin, PassResultSet
from qat.purr.backends.qblox.graph import ControlFlowGraph
from qat.purr.backends.qblox.instructions import EndRepeat, EndSweep
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.instructions import (
    Acquire,
    DeviceUpdate,
    PostProcessing,
    QuantumInstruction,
    Repeat,
    Sweep,
)


class AnalysisPass(PassInfoMixin):
    def run(self, ir, *args, **kwargs):
        pass


class QuantumTargetAnalysis(AnalysisPass):
    def name(self):
        return "QUANTUM_TARGET_ANALYSIS"

    def run(self, builder: InstructionBuilder, *args, **kwargs):
        """
        Collects quantum targets AOT. Useful for subsequent analysis/transform passes
        as well as for backend code generator.
        """
        result = set()
        for inst in builder.instructions:
            if isinstance(inst, QuantumInstruction):
                if isinstance(inst, PostProcessing):
                    for qt in inst.quantum_targets:
                        if isinstance(qt, Acquire):
                            result.update(qt.quantum_targets)
                        else:
                            result.add(qt)
                else:
                    result.update(inst.quantum_targets)
        return PassResultSet((hash(builder), self.name(), result))


class IterBoundsAnalysis(AnalysisPass):
    def name(self):
        return "ITER_BOUNDS"

    def run(self, builder: InstructionBuilder, *args, **kwargs):
        """
        Analyses loop bounds from given value if it's linearly and evenly
        spaced or fails otherwise.
        """
        result = {}
        for inst in builder.instructions:
            if isinstance(inst, Sweep):
                name, sweep_value = next(iter(inst.variables.items()))

                value = sweep_value.value
                if value is None:
                    raise ValueError(f"Cannot process value {value}")

                if isinstance(value, np.ndarray):
                    value = value.tolist()

                if not value:
                    raise ValueError(f"Cannot process value {value}")

                start = value[0]
                step = 0
                end = value[-1]
                count = len(value)

                if count >= 2:
                    step = value[1] - value[0]

                if not np.isclose(step, (end - start) / (count - 1)):
                    raise ValueError(f"Not a regularly partitioned space {value}")

                result[inst] = start, step, end, count
            elif isinstance(inst, Repeat):
                start = 0
                step = 1
                end = inst.repeat_count
                count = inst.repeat_count
                result[inst] = start, step, end, count

            return PassResultSet((hash(builder), self.name(), result))


class CFGAnalysis(AnalysisPass):
    def name(self):
        return "CFG"

    def run(self, builder: InstructionBuilder, *args, **kwargs):
        result = ControlFlowGraph()
        self._build_cfg(builder, result)
        return PassResultSet((hash(builder), self.name(), result))

    def _build_cfg(self, builder: InstructionBuilder, cfg: ControlFlowGraph):
        """
        Recursively (re)discovers (new) header nodes and flow information.
        """

        flow = [(e.src.head(), e.dest.head()) for e in cfg.edges]
        headers = sorted([n.head() for n in cfg.nodes]) or [
            i
            for i, inst in enumerate(builder.instructions)
            if isinstance(inst, (Sweep, Repeat, EndSweep, EndRepeat))
        ]
        if headers[0] != 0:
            headers.insert(0, 0)

        next_flow = set(flow)
        next_headers = set(headers)
        for i, h in enumerate(headers):
            inst_at_h = builder.instructions[h]
            src = cfg.get_or_create_node(h)
            if isinstance(inst_at_h, Repeat):
                next_headers.add(h + 1)
                next_flow.add((h, h + 1))
                dest = cfg.get_or_create_node(h + 1)
                cfg.get_or_create_edge(src, dest)
            elif isinstance(inst_at_h, Sweep):
                s = next(
                    (
                        s
                        for s, inst in enumerate(builder.instructions[h + 1 :])
                        if not isinstance(inst, DeviceUpdate)
                    )
                )
                next_headers.add(s + h + 1)
                next_flow.add((h, s + h + 1))
                src.indices.extend(range(src.tail() + 1, s + h + 1))
                dest = cfg.get_or_create_node(s + h + 1)
                cfg.get_or_create_edge(src, dest)
            elif isinstance(inst_at_h, (EndSweep, EndRepeat)):
                if h < len(builder.instructions) - 1:
                    next_headers.add(h + 1)
                    next_flow.add((h, h + 1))
                    dest = cfg.get_or_create_node(h + 1)
                    cfg.get_or_create_edge(src, dest)
                type = Sweep if isinstance(inst_at_h, EndSweep) else Repeat
                p = next(
                    (p for p in headers[i::-1] if isinstance(builder.instructions[p], type))
                )
                next_headers.add(p)
                next_flow.add((h, p))
                dest = cfg.get_or_create_node(p)
                cfg.get_or_create_edge(src, dest)
            else:
                k = next(
                    (
                        s
                        for s, inst in enumerate(builder.instructions[h + 1 :])
                        if isinstance(inst, (Sweep, Repeat, EndSweep, EndRepeat))
                    ),
                    None,
                )
                if k:
                    next_headers.add(k + h + 1)
                    next_flow.add((h, k + h + 1))
                    src.indices.extend(range(src.tail() + 1, k + h + 1))
                    dest = cfg.get_or_create_node(k + h + 1)
                    cfg.get_or_create_edge(src, dest)

        if next_headers == set(headers) and next_flow == set(flow):
            return

        self._build_cfg(builder, cfg)


class CtrlHwAnalysis(AnalysisPass):
    """
    Perform analyses such as:
    - Lowerability: What can be run on a control hardware stack
    - Batching: After all loop analysis, how many levels or a loop nest can run on the control hardware
    """

    def name(self):
        return "CTRL_HW"

    def run(self, builder: InstructionBuilder, *args, **kwargs):
        pass


class TimelineAnalysis(AnalysisPass):
    """
    Performs analyses necessary for dynamic allocation of control hardware resources to logical channels.
    Loosely speaking, it aims at understanding when exactly instructions are invoked (with **full awareness**
    of control flow (especially loops)) and whether prior resources could be freed up.
    """

    def name(self):
        return "TIMELINE"

    def run(self, builder: InstructionBuilder, *args, **kwargs):
        pass

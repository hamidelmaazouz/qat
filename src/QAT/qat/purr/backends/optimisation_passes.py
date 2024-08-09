from qat.purr.backends.codegen import CodegenPassRegistry
from qat.purr.backends.qblox.instructions import EndRepeat, EndSweep
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.instructions import Repeat, Sweep, SweepValue
from qat.purr.compiler.passbase import TransformPass
from qat.purr.utils.algorithm import stable_partition


class SweepDecomposition(TransformPass):
    def id(self):
        return CodegenPassRegistry.SWEEP_DECOMPOSITION

    def do_run(self, builder: InstructionBuilder, *args, **kwargs):
        """
        Decomposes complex multi-dim sweeps into simpler one-dim sweeps.
        """
        result = []
        for i, inst in enumerate(builder.instructions):
            if isinstance(inst, Sweep) and len(inst.variables) > 1:
                for name, value in inst.variables.items():
                    result.append(Sweep(SweepValue(name, value)))
            else:
                result.append(inst)
        builder.instructions = result


class ScopeBalancing(TransformPass):
    def id(self):
        return CodegenPassRegistry.SCOPE_BALANCING

    def do_run(self, builder: InstructionBuilder, *args):
        """
        Bubbles up all sweeps and repeats to the beginning of the list.
        Adds delimiter instructions to the repeats and sweeps signifying the end of their scopes.
        Collects targets AOT.

        Intended for legacy existing builders and the relative order of instructions guarantees backwards
        compatibility.
        """

        head, tail = stable_partition(
            builder.instructions, lambda inst: isinstance(inst, (Sweep, Repeat))
        )

        delimiters = [
            EndSweep() if isinstance(inst, Sweep) else EndRepeat() for inst in head
        ]

        builder.instructions = head + tail + delimiters[::-1]

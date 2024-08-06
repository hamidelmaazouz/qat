from abc import abstractmethod

from qat.purr.compiler.ir.pass_base import PassInfoMixin, PassResultSet
from qat.purr.backends.qblox.instructions import EndRepeat, EndSweep
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.instructions import Repeat, Sweep, SweepValue
from qat.purr.compiler.ir.pass_pipeline import PassRegistry
from qat.purr.utils.algorithm import stable_partition


class TransformPass(PassInfoMixin):
    def run(self, ir, *args, **kwargs):
        self.do_run(ir, *args, **kwargs)
        return PassResultSet()

    @abstractmethod
    def do_run(self, ir, *args, **kwargs):
        pass


class SweepDecomposition(TransformPass):
    def id(self):
        return PassRegistry.SWEEP_DECOMPOSITION

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
        return PassRegistry.SCOPE_BALANCING

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

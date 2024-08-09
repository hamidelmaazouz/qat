from typing import Set

import numpy as np
import pytest

from qat.purr.backends.analysis_passes import (
    CFGAnalysis,
    IterBoundsAnalysis,
    QuantumTargetAnalysis,
)
from qat.purr.backends.codegen import CodegenPassRegistry
from qat.purr.backends.optimisation_passes import ScopeBalancing, SweepDecomposition
from qat.purr.backends.qblox.instructions import EndRepeat, EndSweep
from qat.purr.backends.verification_passes import ScopeBalanceValidation
from qat.purr.compiler.instructions import Repeat, Sweep
from src.tests.qblox.builder_nuggets import resonator_spect
from src.tests.qblox.utils import ClusterInfo

count = 100
cases = [([1, 2, 4, 10], None), ([-0.1, 0, 0.1, 0.2, 0.3], (-0.1, 0.1, 0.3, 5))] + [
    (np.linspace(b[0], b[1], count), (b[0], 1, b[1], count)) for b in [(1, 100), (-50, 49)]
]


@pytest.mark.parametrize("value, bounds", cases)
def test_extract_iter_bounds(value, bounds):
    if bounds is None:
        with pytest.raises(ValueError):
            IterBoundsAnalysis.extract_iter_bounds(value)
    else:
        assert IterBoundsAnalysis.extract_iter_bounds(value) == bounds


@pytest.mark.parametrize("model", [ClusterInfo], indirect=True)
class TestPassPipeline:
    def test_quantum_target_analysis(self, model):
        qubit_index = 0
        qubit = model.get_qubit(qubit_index)
        builder = resonator_spect(model, qubit_index)

        rs = QuantumTargetAnalysis().run(builder)
        assert len(rs.data) == 1
        rk, value = next(iter(rs.data.items()))
        assert rk.ir_id == hash(builder)
        assert rk.pass_id == CodegenPassRegistry.QUANTUM_TARGETS
        assert isinstance(value, Set)
        assert len(value) == 3
        assert qubit.get_drive_channel() in value
        assert qubit.get_measure_channel() in value
        assert qubit.get_acquire_channel() in value

    def test_sweep_decomposition(self, model):
        builder = resonator_spect(model)

        sweeps = [s for s in builder.instructions if isinstance(s, Sweep)]
        expected = sum([len(s.variables) for s in sweeps], 0)
        SweepDecomposition().run(builder)
        assert len([s for s in builder.instructions if isinstance(s, Sweep)]) == expected

    def test_scope_balancing(self, model):
        builder = resonator_spect(model)

        sweeps = [inst for inst in builder.instructions if isinstance(inst, Sweep)]
        end_sweeps = [inst for inst in builder.instructions if isinstance(inst, EndSweep)]
        repeats = [inst for inst in builder.instructions if isinstance(inst, Repeat)]
        end_repeats = [inst for inst in builder.instructions if isinstance(inst, EndRepeat)]
        assert len(sweeps) == 1
        assert len(end_sweeps) == 0
        assert len(repeats) == 1
        assert len(end_repeats) == 0

        with pytest.raises(ValueError):
            ScopeBalanceValidation().run(builder)

        ScopeBalancing().run(builder)
        sweeps = [inst for inst in builder.instructions if isinstance(inst, Sweep)]
        end_sweeps = [inst for inst in builder.instructions if isinstance(inst, EndSweep)]
        repeats = [inst for inst in builder.instructions if isinstance(inst, Repeat)]
        end_repeats = [inst for inst in builder.instructions if isinstance(inst, EndRepeat)]
        assert len(sweeps) == 1
        assert len(end_sweeps) == 1
        assert len(repeats) == 1
        assert len(end_repeats) == 1

        ScopeBalanceValidation().run(builder)

    def test_cfg_analysis(self, model):
        builder = resonator_spect(model)

        ScopeBalancing().run(builder)
        rs = CFGAnalysis().run(builder)
        assert len(rs.data) == 1
        rk, value = next(iter(rs.data.items()))
        assert rk.ir_id == hash(builder)
        assert rk.pass_id == CodegenPassRegistry.CFG
        assert len(value.nodes) == 5
        assert len(value.edges) == 6

    def test_iter_bounds_analysis(self, model):
        builder = resonator_spect(model)

        analysis = IterBoundsAnalysis()
        analysis.run(builder)

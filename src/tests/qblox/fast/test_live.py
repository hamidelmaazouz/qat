import pytest

from qat.purr.backends.qblox.fast.live import FastQbloxLiveEngine
from src.tests.qblox.builder_nuggets import resonator_spect
from src.tests.qblox.utils import ClusterInfo


@pytest.mark.parametrize("model", [ClusterInfo()], indirect=True)
class TestFastQbloxLiveEngine:
    def test_execute(self, model):
        builder = resonator_spect(model)
        engine = FastQbloxLiveEngine(model)
        model.control_hardware.dump_sequence = True
        results = engine.execute(builder.instructions)
        assert results is not None

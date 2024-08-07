import pytest

from qat.purr.backends.qblox.fast.codegen import FastQbloxEmitter
from qat.purr.backends.qblox.fast.live import FastQbloxLiveEngine
from qat.purr.utils.logger import get_default_logger
from src.tests.qblox.builder_nuggets import resonator_spect

log = get_default_logger()


@pytest.mark.parametrize("model", [None], indirect=True)
class TestFastQbloxEmitter:
    def test_emit_packages(self, model):
        builder = resonator_spect(model)
        engine = FastQbloxLiveEngine(model)
        instructions = engine.run_pass_pipeline(builder)
        packages = FastQbloxEmitter(instructions).emit_packages()
        assert packages is not None

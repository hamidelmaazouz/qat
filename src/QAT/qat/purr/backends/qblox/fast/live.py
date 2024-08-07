from typing import Dict, List

import numpy as np

from qat.purr.backends.qblox.fast.codegen import FastQbloxEmitter
from qat.purr.backends.qblox.live import QbloxLiveEngine
from qat.purr.compiler.instructions import Instruction
from qat.purr.compiler.interrupt import Interrupt, NullInterrupt
from qat.purr.utils.logger import get_default_logger
from qat.purr.utils.logging_utils import log_duration

log = get_default_logger()


class FastQbloxLiveEngine(QbloxLiveEngine):

    def optimize(self, instructions):
        pass

    def validate(self, instructions: List[Instruction]):
        pass

    def run_pass_pipeline(self, builder):
        pass_manager = self.model.build_pass_pipeline()
        return pass_manager.run(builder)

    def _common_execute(self, builder, interrupt: Interrupt = NullInterrupt()):
        """Executes this qat file against this current hardware."""
        self._model_exists()

        analyses = self.run_pass_pipeline(builder)

        with log_duration("QPU returned results in {} seconds."):
            packages = FastQbloxEmitter(analyses).emit_packages()
            self.model.control_hardware.set_data(packages)
            playback_results: Dict[str, np.ndarray] = (
                self.model.control_hardware.start_playback(None, None)
            )

            # Process metadata assign/return values to make sure the data is in the
            # right form.
            # results = self._process_results(results, qat_file)
            # results = self._process_assigns(results, qat_file)

            # return results

            return playback_results

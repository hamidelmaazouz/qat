from abc import abstractmethod
from typing import Dict, List

import numpy as np

from qat.purr.backends.analysis_passes import QuantumTargetAnalysis, CFGAnalysis
from qat.purr.backends.optimisation_passes import SweepDecomposition, ScopeBalancing
from qat.purr.backends.qblox.fast.codegen import FastQbloxEmitter
from qat.purr.backends.qblox.live import QbloxLiveEngine
from qat.purr.backends.verification_passes import ScopeBalanceValidation
from qat.purr.compiler.instructions import Instruction
from qat.purr.compiler.interrupt import Interrupt, NullInterrupt
from qat.purr.compiler.passbase import PassManager
from qat.purr.utils.logger import get_default_logger
from qat.purr.utils.logging_utils import log_duration

log = get_default_logger()


class EngineMixin:
    @abstractmethod
    def build_pass_pipeline(self):
        pass


class FastQbloxLiveEngine(QbloxLiveEngine, EngineMixin):

    def optimize(self, instructions):
        pass

    def validate(self, instructions: List[Instruction]):
        pass

    def build_pass_pipeline(self):
        pm = PassManager()
        pm.add(QuantumTargetAnalysis())
        pm.add(SweepDecomposition())
        pm.add(ScopeBalancing())
        pm.add(ScopeBalanceValidation())
        pm.add(CFGAnalysis())
        return pm

    def _common_execute(self, builder, interrupt: Interrupt = NullInterrupt()):
        """Executes this qat file against this current hardware."""
        self._model_exists()

        pm = self.build_pass_pipeline()
        pass_rs = pm.run(builder)

        with log_duration("QPU returned results in {} seconds."):
            packages = FastQbloxEmitter(pass_rs).emit_packages()
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

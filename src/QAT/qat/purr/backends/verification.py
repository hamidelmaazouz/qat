# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Oxford Quantum Circuits Ltd

from abc import ABC

from qat.purr.backends.live import LiveHardwareModel
from qat.purr.compiler.emitter import QatFile
from qat.purr.compiler.execution import QuantumExecutionEngine


class QPUVersion:
    def __init__(self, make: str, version: str = None):
        self.make = make
        self.version = version

    def __repr__(self):
        return f"{self.make}-{self.version or 'latest'}"


class Lucy:
    # TODO: Automate this from class name, not trivially doable due to the class still being
    #  initialized at this point. Don't want to do deferred exe for such a simple case.
    _name = "Lucy"

    Latest = QPUVersion(_name)


class VerificationModel(LiveHardwareModel):
    def __init__(self, qpu_version, verification_engine_type: type):
        super().__init__(None, [verification_engine_type], None)
        self.version = qpu_version


def get_verification_model(qpu_type: QPUVersion):
    """
    Get verification model for a particular QPU make and model. Each make has its own class, which has a field
    that is each individual version available for verification.

    For example, if you wanted to verify our Lucy machine, that'd be done with:
    ``
    get_verification_model(Lucy.Latest)
    ``

    Or with a specific version:
    ``
    get_verification_model(Lucy.XY)
    ``
    """
    if not isinstance(qpu_type, QPUVersion):
        raise ValueError(f"{qpu_type} is not a QPU version, can't find verification engine.")

    if qpu_type.make == Lucy.__name__:
        # TODO: Should have an apply_setup_to_hardware in live which people can use to build our own architecture.
        model = VerificationModel(qpu_type, LucyVerificationEngine)
        raise NotImplementedError("No lucy-specific model created yet.")

    return None


class VerificationEngine(QuantumExecutionEngine, ABC):
    ...


class LucyVerificationEngine(VerificationEngine):
    def _execute_on_hardware(self, sweep_iterator, package: QatFile):
        # TODO: Add actual verification
        raise NotImplementedError("No lucy-specific verification yet.")

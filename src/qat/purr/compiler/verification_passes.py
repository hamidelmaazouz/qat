from numbers import Number
from typing import List

import numpy as np

from qat.ir.pass_base import ValidationPass
from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.devices import MaxPulseLength, PulseChannel, Qubit
from qat.purr.compiler.instructions import (
    Acquire,
    AcquireMode,
    CustomPulse,
    PostProcessing,
    ProcessAxis,
    Pulse,
    Sweep,
    Variable,
)


class CtrlHwValidation(ValidationPass):
    """
    Extracted from QuantumExecutionEngine.validate()
    Better pass name/id ?
    """

    def __init__(self, engine):
        # TODO - Move max_instruction_len to the model / control hardware
        self.max_instruction_len = engine.max_instruction_len

    def do_run(self, builder: InstructionBuilder, *args, **kwargs):
        instruction_length = len(builder.instructions)
        if instruction_length > self.max_instruction_len:
            raise ValueError(
                f"Program too large to be run in a single block on current hardware. "
                f"{instruction_length} instructions."
            )

        for inst in builder.instructions:
            if isinstance(inst, Acquire) and not inst.channel.acquire_allowed:
                raise ValueError(
                    f"Cannot perform an acquire on the physical channel with id "
                    f"{inst.channel.physical_channel}"
                )
            if isinstance(inst, (Pulse, CustomPulse)):
                duration = inst.duration
                if isinstance(duration, Number) and duration > MaxPulseLength:
                    raise ValueError(
                        f"Max Waveform width is {MaxPulseLength} s "
                        f"given: {inst.duration} s"
                    )
                elif isinstance(duration, Variable):
                    values = next(
                        iter(
                            [
                                sw.variables[duration.name]
                                for sw in builder.instructions
                                if isinstance(sw, Sweep)
                                and duration.name in sw.variables.keys()
                            ]
                        )
                    )
                    if np.max(values) > MaxPulseLength:
                        raise ValueError(
                            f"Max Waveform width is {MaxPulseLength} s "
                            f"given: {values} s"
                        )


class PostProcessingValidation(ValidationPass):
    def __init__(self, model):
        self.model = model

    def do_run(self, builder: InstructionBuilder, *args, **kwargs):
        consumed_qubits: List[str] = []
        for inst in builder.instructions:
            if isinstance(inst, PostProcessing):
                if (
                    inst.acquire.mode == AcquireMode.SCOPE
                    and ProcessAxis.SEQUENCE in inst.axes
                ):
                    raise ValueError(
                        "Invalid post-processing! Post-processing over SEQUENCE is "
                        "not possible after the result is returned from hardware "
                        "in SCOPE mode!"
                    )
                elif (
                    inst.acquire.mode == AcquireMode.INTEGRATOR
                    and ProcessAxis.TIME in inst.axes
                ):
                    raise ValueError(
                        "Invalid post-processing! Post-processing over TIME is not "
                        "possible after the result is returned from hardware in "
                        "INTEGRATOR mode!"
                    )
                elif inst.acquire.mode == AcquireMode.RAW:
                    raise ValueError(
                        "Invalid acquire mode! The live hardware doesn't support "
                        "RAW acquire mode!"
                    )

            # Check if we've got a measure in the middle of the circuit somewhere.
            elif isinstance(inst, Acquire):
                for qbit in self.model.qubits:
                    if qbit.get_measure_channel() == inst.channel:
                        consumed_qubits.append(qbit)
            elif isinstance(inst, Pulse):
                # Find target qubit from instruction and check whether it's been
                # measured already.
                acquired_qubits = [
                    self.model._resolve_qb_pulse_channel(chanbit)[0] in consumed_qubits
                    for chanbit in inst.quantum_targets
                    if isinstance(chanbit, (Qubit, PulseChannel))
                ]

                if any(acquired_qubits):
                    raise ValueError(
                        "Mid-circuit measurements currently unable to be used."
                    )
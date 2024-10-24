import numpy as np

from qat.purr.compiler.devices import PulseShapeType
from qat.purr.compiler.instructions import SweepValue, Variable
from qat.purr.compiler.runtime import get_builder


def resonator_spect(model, qubit_indices=None):
    qubit_indices = qubit_indices or [0]
    builder = get_builder(model)
    num_points = 10
    freq_range = 50e6
    for index in qubit_indices:
        qubit = model.get_qubit(index)
        measure_channel = qubit.get_measure_channel()
        acquire_channel = qubit.get_acquire_channel()

        center_freq = qubit.get_acquire_channel().frequency
        freqs = center_freq + np.linspace(-freq_range, freq_range, num_points)
        var_name = f"freq{qubit.index}"
        output_variable = f"Q{qubit.index}"

        builder = (
            builder.sweep(SweepValue(var_name, freqs))
            .device_assign(measure_channel, "frequency", Variable(var_name))
            .device_assign(acquire_channel, "frequency", Variable(var_name))
            .measure_mean_signal(qubit, output_variable)
            .repeat(1000, 500e-6)
        )
    return builder


def qubit_spect(model, qubit_indices=None):
    qubit_indices = qubit_indices or [0]
    builder = get_builder(model)
    num_points = 10
    freq_range = 50e6
    drive_amp_dbm = -40

    for index in qubit_indices:
        qubit = model.get_qubit(index)
        drive_channel = qubit.get_drive_channel()

        freqs = drive_channel.frequency + np.linspace(-freq_range, freq_range, num_points)
        drive_amp_v = np.sqrt(10 ** (((drive_amp_dbm + 12) / 10) - 1))
        var_name = f"freq{qubit.index}"
        output_variable = f"Q{qubit.index}"

        builder = (
            builder.device_assign(drive_channel, "scale", 1)
            .sweep(SweepValue(var_name, freqs))
            .device_assign(drive_channel, "frequency", Variable(var_name))
            .pulse(
                drive_channel,
                PulseShapeType.SQUARE,
                width=80e-6,
                amp=drive_amp_v,
                phase=0.0,
                drag=0.0,
                rise=1.0 / 3.0,
            )
            .measure_mean_signal(qubit, output_variable=output_variable)
        )
    return builder

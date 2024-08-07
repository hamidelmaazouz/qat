from qat.purr.compiler.builders import InstructionBuilder
from qat.purr.compiler.runtime import QuantumRuntime


class QbloxQuantumRuntime(QuantumRuntime):
    """
    Temporary object to show the new pass pipeline. Eng goal is to adopt it wider in the frontend/middle-end.

    Notice how polymorphic calls to optimize() and validate() are avoided. Instead, we have a flat structure
    of passes that are tailored for the hw.

    Any new pass can be easily plugged without the need for inheritance.
    """

    def _common_execute(
        self,
        fexecute: callable,
        ir,
        results_format=None,
        repeats=None,
        error_mitigation=None,
    ):
        if self.engine is None:
            raise ValueError("No execution engine available.")

        if not isinstance(ir, InstructionBuilder):
            raise ValueError(
                f"Invalid IR. Expected InstructionBuilder, got {type(ir)} instead"
            )

        ir = self.engine.optimize(ir)
        self.engine.validate(ir)
        self.record_metric(MetricsType.OptimizedInstructionCount, opt_inst_count := len(ir))
        log.info(f"Optimized instruction count: {opt_inst_count}")

        results = fexecute(ir)
        results = self._transform_results(results, results_format, repeats)
        return self._apply_error_mitigation(results, ir, error_mitigation)

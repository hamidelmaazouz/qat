from qat.purr.backends.qblox.instructions import EndRepeat, EndSweep
from qat.purr.backends.validation import ValidationPass
from qat.purr.compiler.instructions import Repeat, Sweep


class ScopeBalanceValidation(ValidationPass):
    def run(self, builder, *args, **kwargs):
        """
        Repeat and Sweep scopes are valid if they have a start and end delimiters and if the delimiters
        are balanced.
        """

        stack = []
        for inst in builder.instructions:
            if isinstance(inst, (Sweep, Repeat)):
                stack.append(inst)
            elif isinstance(inst, (EndSweep, EndRepeat)):
                type = Sweep if isinstance(inst, EndSweep) else Repeat
                try:
                    if not isinstance(stack.pop(), type):
                        raise ValueError(f"Unbalanced {type} scope. Found orphan {inst}")
                except IndexError:
                    raise ValueError(f"Unbalanced {type} scope. Found orphan {inst}")

        if stack:
            raise ValueError(f"Unbalanced scopes. Found orphans {stack}")

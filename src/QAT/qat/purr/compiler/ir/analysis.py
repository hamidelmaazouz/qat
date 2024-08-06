from dataclasses import dataclass
from typing import Any, List, Union

import numpy as np

from qat.purr.compiler.ir.pass_base import PassInfoMixin
from qat.purr.utils.logger import get_default_logger

log = get_default_logger()


class AnalysisPass(PassInfoMixin):
    def run(self, ir, *args, **kwargs):
        pass

"""
SplitWeave: A DSL for 2D pattern programs.

Used for synthetic pattern generation and analogical quartets (Pattern Analogy project).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("splitweave")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Subpackages
from . import symbolic
from . import torch_compute
from . import cfg_to_sw

# High-level API: expression -> pattern
from .torch_compute import (
    grid_eval,
    rec_grid_eval,
    aa_eval,
    evaluate_pattern,
    evaluate_pattern_expr,
    rec_eval_pattern_expr,
    sws_to_fn_mapper,
)
from .cfg_to_sw.parser import mtp_config_to_expr
from .cfg_to_sw.evaluate import evaluate_pattern_cfg

__all__ = [
    "__version__",
    "symbolic",
    "torch_compute",
    "cfg_to_sw",
    "grid_eval",
    "rec_grid_eval",
    "aa_eval",
    "evaluate_pattern",
    "evaluate_pattern_expr",
    "rec_eval_pattern_expr",
    "sws_to_fn_mapper",
    "mtp_config_to_expr",
    "evaluate_pattern_cfg",
]

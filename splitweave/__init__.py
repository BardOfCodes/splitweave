"""
SplitWeave: A DSL for 2D pattern programs.

Provides a unified symbolic language for Multi-Tile Patterns (MTP)
and Random Split Patterns (RSP), used for synthetic pattern generation
and analogical quartets (Pattern Analogy project, CVPR 2025).
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

# Config -> expression -> pattern
from .cfg_to_sw.parser import mtp_config_to_expr, rsp_config_to_expr
from .cfg_to_sw.evaluate import evaluate_pattern_cfg, evaluate_rsp_cfg

__all__ = [
    "__version__",
    # subpackages
    "symbolic",
    "torch_compute",
    "cfg_to_sw",
    # evaluation
    "grid_eval",
    "rec_grid_eval",
    "aa_eval",
    "evaluate_pattern",
    "evaluate_pattern_expr",
    "rec_eval_pattern_expr",
    "sws_to_fn_mapper",
    # config-to-expression
    "mtp_config_to_expr",
    "rsp_config_to_expr",
    "evaluate_pattern_cfg",
    "evaluate_rsp_cfg",
]

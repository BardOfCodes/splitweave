"""
Bridge from pattern configs to splitweave expressions.

This package provides:

- Parsers that convert yacs ``CfgNode`` configs into
  splitweave symbolic expressions (see ``parser.py``).
- Convenience functions that run the full pipeline:
  config -> load tiles -> build expression -> evaluate (see ``evaluate.py``).

Public API
----------
mtp_config_to_expr : CfgNode -> Expr
    Convert an MTP config to a pattern expression.
rsp_config_to_expr : CfgNode -> Expr
    Convert an RSP config to a pattern expression.
load_tile_tensors_from_config : CfgNode -> dict[str, Tensor]
    Load tile images from disk into tensors.
evaluate_pattern_cfg : CfgNode, Sketcher -> (canvas, grid_ids)
    End-to-end evaluation (auto-detects MTP vs RSP).
evaluate_rsp_cfg : CfgNode, Sketcher -> (canvas, grid_ids)
    End-to-end evaluation for RSP configs.
"""

from .parser import mtp_config_to_expr, rsp_config_to_expr, load_tile_tensors_from_config
from .evaluate import evaluate_pattern_cfg, evaluate_rsp_cfg

__all__ = [
    "mtp_config_to_expr",
    "rsp_config_to_expr",
    "load_tile_tensors_from_config",
    "evaluate_pattern_cfg",
    "evaluate_rsp_cfg",
]

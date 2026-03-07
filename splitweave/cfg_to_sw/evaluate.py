"""
Patternator->splitweave bridge: CFG -> single expression -> pattern.

Consistent pipeline for both MTP and RSP:
  1. Generate config: sample_mtp_pattern / sample_rsp_pattern
  2. Load tiles:      load_tile_tensors_from_config(cfg.tile_cfg, tile_dir, device)
  3. Build expr:      mtp_config_to_expr / rsp_config_to_expr (cfg, tile_tensors=..., device=...)
  4. Execute:         evaluate_pattern(expr, sketcher) -> (canvas, grid_ids)

Convenience:
  evaluate_pattern_cfg(cfg, sketcher, tile_dir=...) does steps 2-4.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch as th
from yacs.config import CfgNode as CN

from splitweave.torch_compute import evaluate_pattern

from .parser import mtp_config_to_expr, rsp_config_to_expr, load_tile_tensors_from_config


def _is_rsp_config(cfg: CN) -> bool:
    """True if cfg has RSP structure (layout + fill). MTP uses layout_cfg instead."""
    return hasattr(cfg, "layout") and hasattr(cfg, "fill")


def evaluate_rsp_cfg(
    cfg: CN,
    sketcher,
    *,
    tile_dir: Optional[str] = None,
    tile_tensors: Optional[Dict[str, th.Tensor]] = None,
    aa: int = 1,
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Evaluate an RSP config: load tiles -> rsp_config_to_expr -> evaluate_pattern.
    """
    device = str(sketcher.device)
    if tile_tensors is None and tile_dir is not None and hasattr(cfg, "tile_cfg"):
        tile_tensors = load_tile_tensors_from_config(cfg.tile_cfg, tile_dir, device=device)
    expr = rsp_config_to_expr(cfg, tile_tensors=tile_tensors, device=device)
    canvas, grid_ids = evaluate_pattern(expr, sketcher, aa=aa)
    canvas = th.nan_to_num(canvas, nan=0.0, posinf=1.0, neginf=0.0)
    return canvas, grid_ids


def evaluate_pattern_cfg(
    cfg: CN,
    sketcher,
    *,
    tile_dir: Optional[str] = None,
    tile_tensors: Optional[Dict[str, th.Tensor]] = None,
    aa: int = 1,
) -> Tuple[th.Tensor, th.Tensor]:
    """
    Evaluate a patternator config: CFG -> single expression -> execute -> (canvas, grid_ids).

    Parameters
    ----------
    cfg:
        Patternator config (as returned by sample_mtp_pattern or sample_default_mtp_pattern).
    sketcher:
        GeoLIPI Sketcher used by splitweave and geolipi evaluators.
    tile_dir:
        Directory to load tile images from. Required if tile_tensors is not provided.
    tile_tensors:
        Optional preloaded dict (tilefile path -> tensor [H, W, C]). If provided, tile_dir is ignored.
    aa:
        Super-sampling anti-aliasing factor (passed through to evaluate_pattern).

    Returns
    -------
    canvas:
        Final RGBA canvas tensor of shape [R*R, 4] or [R, R, 4].
    grid_ids:
        Grid cell IDs from the layout expression.
    """
    device = str(sketcher.device)
    if tile_tensors is None and tile_dir is not None and hasattr(cfg, "tile_cfg"):
        tile_tensors = load_tile_tensors_from_config(cfg.tile_cfg, tile_dir, device=device)
    if _is_rsp_config(cfg):
        expr = rsp_config_to_expr(cfg, tile_tensors=tile_tensors, device=device)
    else:
        expr = mtp_config_to_expr(
            cfg, tile_dir=tile_dir, tile_tensors=tile_tensors,
            device=device, resolution=getattr(sketcher, "resolution", None),
        )
    canvas, grid_ids = evaluate_pattern(expr, sketcher, aa=aa)
    canvas = th.nan_to_num(canvas, nan=0.0, posinf=1.0, neginf=0.0)
    return canvas, grid_ids


"""
Recursive grid evaluator for splitweave expressions.

Uses singledispatch on expression types to evaluate the symbolic
expression tree into (grid, grid_ids) tensors.

Pattern: follows geolipi's evaluate_expression.py single-dispatch style.

Core grid dispatch handlers live here. Additional handlers are
registered by sub-modules imported in __init__.py:
  - eval_deformations.py  -- DeformGrid handlers
  - eval_signals.py       -- Signal2D handlers
  - eval_sympy.py         -- GLExpr / AssocOp / Pow / leaf types
  - eval_cell_effects.py  -- Layout cell effect handlers
"""
import sys
from typing import Tuple, Optional

import sympy as sp
import torch as th
import geolipi.symbolic as gls
from geolipi.torch_compute import Sketcher
from geolipi.torch_compute import recursive_evaluate
from geolipi.symbolic.symbol_types import SUPERSET_TYPE

from .mapping import sws_to_fn_mapper
import splitweave.symbolic as sws

if sys.version_info >= (3, 11):
    from functools import singledispatch
else:
    from geolipi.torch_compute.patched_functools import singledispatch


# ============================================================
# Helpers
# ============================================================

def _ensure_2d_ids(ids):
    """Pad single-channel IDs to 2-channel (N, 2) so every partition
    level contributes at least 2 ID columns."""
    if ids is None:
        return ids
    if ids.shape[-1] == 1:
        return th.cat([ids, th.zeros_like(ids)], dim=-1)
    return ids


def merge_grid_ids(grid_ids, new_ids):
    """Stack new partition IDs onto existing ones along the last dim.

    Each partition level produces a 2D (N, 2) cell ID (after padding).
    Successive partitions are concatenated so grid_ids grows:
    (N, 2) -> (N, 4) -> (N, 6) etc., preserving the full hierarchy.
    """
    new_ids = _ensure_2d_ids(new_ids)
    if grid_ids is None:
        return new_ids
    elif new_ids is None:
        return grid_ids
    else:
        return th.cat([grid_ids, new_ids], dim=-1)


def _unwrap_tensor(t):
    """Unwrap a single-element tensor to a Python scalar."""
    if isinstance(t, th.Tensor) and t.numel() == 1:
        return t.item()
    return t


def _parse_sws_params(expression, params, param_start_index=1):
    """Parse sympy/tensor params from a splitweave expression.

    Uses expression.get_arg(param_start_index + i) to resolve params.
    When params is expression.args[1:], use param_start_index=1; when
    params is expression.args, use param_start_index=0.
    Single-element tensors are unwrapped to Python scalars.
    """
    if params:
        param_list = []
        for i in range(len(params)):
            idx = param_start_index + i
            if idx < len(expression.args):
                cur_param = expression.get_arg(idx)
            else:
                cur_param = params[i]
            if isinstance(cur_param, th.Tensor):
                cur_param = _unwrap_tensor(cur_param)
            elif isinstance(cur_param, sp.Integer):
                cur_param = int(float(cur_param))
            elif isinstance(cur_param, sp.Float):
                cur_param = float(cur_param)
            elif isinstance(cur_param, sp.Tuple):
                if len(cur_param) == 1:
                    cur_param = float(cur_param[0])
                else:
                    cur_param = th.tensor([float(x) for x in cur_param])
            elif isinstance(cur_param, gls.Param):
                cur_param = _unwrap_tensor(cur_param.get_arg(0))
            elif isinstance(cur_param, sp.Symbol):
                cur_param = str(cur_param)
            param_list.append(cur_param)
        params = param_list
    return params


# ============================================================
# Entry point
# ============================================================

def grid_eval(expr: SUPERSET_TYPE, sketcher: Sketcher,
              grid: Optional[th.Tensor] = None,
              grid_ids: Optional[th.Tensor] = None) -> Tuple[th.Tensor, Optional[th.Tensor]]:
    """Evaluate a splitweave expression to produce (grid, grid_ids)."""
    if grid is None:
        grid = sketcher.get_base_coords()
    return rec_grid_eval(expr, sketcher, grid, grid_ids)


# ============================================================
# Single-dispatch evaluator
# ============================================================

@singledispatch
def rec_grid_eval(expr: SUPERSET_TYPE, sketcher: Sketcher,
                  grid=None, grid_ids=None) -> Tuple[th.Tensor, Optional[th.Tensor]]:
    raise NotImplementedError(
        f"Expression type {type(expr)} is not supported for recursive grid evaluation."
    )


# ---- Grid Instantiation ----

@rec_grid_eval.register
def eval_instantiate_grid(expression: sws.InstantiateGrid, sketcher: Sketcher,
                          grid=None, grid_ids=None):
    grid_func = sws_to_fn_mapper[expression.__class__]
    grid = grid_func(grid, 1, 1)
    return grid, grid_ids


# ---- Grid Conversion ----

@rec_grid_eval.register
def eval_convert_grid(expression: sws.ConvertGrid, sketcher: Sketcher,
                      grid=None, grid_ids=None):
    grid, grid_ids = rec_grid_eval(expression.args[0], sketcher, grid, grid_ids)
    grid_func = sws_to_fn_mapper[expression.__class__]
    params = expression.args[1:]
    params = _parse_sws_params(expression, params)
    grid = grid_func(grid, *params)
    return grid, grid_ids


# ---- SDF Ring Partition (RSP shape_sdf / shape_overlay; custom logic) ----

@rec_grid_eval.register
def eval_sdf_ring_partition(expression: sws.SDFRingPartition, sketcher: Sketcher,
                            grid=None, grid_ids=None) -> Tuple[th.Tensor, Optional[th.Tensor]]:
    """Evaluate SDF contour rings: shape_expr -> SDF on grid -> round to ring IDs."""
    grid, grid_ids = rec_grid_eval(expression.args[0], sketcher, grid, grid_ids)
    shape_expr = expression.get_arg(1)
    sdf_step_size = float(expression.get_arg(2))
    ring_counts = int(expression.get_arg(3))
    growth_mode = str(expression.get_arg(4)) if len(expression.args) > 4 else "linear"

    outline = gls.AlphaMask2D(shape_expr)
    base_shape = gls.AlphaToSDF2D(outline, 1.0 / sketcher.resolution)
    sdf = recursive_evaluate(base_shape.tensor(device=str(sketcher.device)), sketcher, coords=grid)
    sdf = sdf.clone()
    sdf[sdf < 0] = 0
    if growth_mode == "logarithmic":
        sdf = th.log(sdf * 5 + 1)
    contours = th.round(sdf / sdf_step_size)
    ring_ids = contours[..., None].long().float()
    ring_ids = th.where(ring_ids > ring_counts, th.full_like(ring_ids, -1.0), ring_ids)
    return grid, ring_ids


# ---- Grid Partitioning ----

@rec_grid_eval.register
def eval_partition_grid(expression: sws.PartitionGrid, sketcher: Sketcher,
                        grid=None, grid_ids=None):
    grid, grid_ids = rec_grid_eval(expression.args[0], sketcher, grid, grid_ids)
    grid_func = sws_to_fn_mapper[expression.__class__]
    params = expression.args[1:]
    params = _parse_sws_params(expression, params)

    if isinstance(expression, (sws.IrregularRectRepeat, sws.IrregularRectRepeatEdge)):
        simple_grid = sketcher.get_base_coords()
        grid, new_ids = grid_func(grid, simple_grid, *params)
    else:
        grid, new_ids = grid_func(grid, *params)

    new_ids = _ensure_2d_ids(new_ids)
    grid_ids = merge_grid_ids(grid_ids, new_ids)
    return grid, grid_ids


# ---- Grid Transforms (constant) ----

@rec_grid_eval.register
def eval_transform_grid(expression: sws.TransformGrid, sketcher: Sketcher,
                        grid=None, grid_ids=None):
    grid, grid_ids = rec_grid_eval(expression.args[0], sketcher, grid, grid_ids)
    grid_func = sws_to_fn_mapper[expression.__class__]
    params = expression.args[1:]
    params = _parse_sws_params(expression, params)
    grid = grid_func(grid, *params)
    return grid, grid_ids


# ---- Signal-driven Transforms ----

@rec_grid_eval.register
def eval_signal_transform_grid(expression: sws.SignalTransformGrid, sketcher: Sketcher,
                               grid=None, grid_ids=None):
    grid, grid_ids = rec_grid_eval(expression.args[0], sketcher, grid, grid_ids)
    signal_expr = expression.args[1]
    if isinstance(signal_expr, gls.GLExpr):
        signal, _ = rec_grid_eval(signal_expr, sketcher, grid, grid_ids)
    elif isinstance(signal_expr, gls.Param):
        signal = signal_expr.get_arg(0)
    elif isinstance(signal_expr, gls.GLFunction):
        signal = recursive_evaluate(signal_expr, sketcher)
    else:
        raise NotImplementedError(f"Cannot evaluate signal: {signal_expr}")
    grid_func = sws_to_fn_mapper[expression.__class__]
    grid = grid_func(grid, signal)
    return grid, grid_ids


# ---- Scalar 2D (noise generators) ----

@rec_grid_eval.register
def eval_scalar_2d(expression: sws.Scalar2D, sketcher: Sketcher,
                   grid=None, grid_ids=None):
    grid_func = sws_to_fn_mapper[expression.__class__]
    resolution = sketcher.resolution
    params = expression.args
    params = _parse_sws_params(expression, params, param_start_index=0)
    fraction = resolution / params[0]
    if not isinstance(fraction, th.Tensor):
        fraction = th.tensor(fraction, device=sketcher.device)
    grid = grid_func(resolution, fraction)
    return grid, grid_ids

"""
Layout cell effect dispatch handlers (Translate, Rotate, Scale, Reflect).

These modify the grid before tile evaluation. Each takes (layout_expr, signal_expr, ...)
and returns (modified_grid, grid_ids).
"""
import torch as th

from geolipi.torch_compute import Sketcher
import splitweave.symbolic as sws

from .eval_grid import rec_grid_eval, grid_eval, _parse_sws_params
from .eval_cell_canvas import process_signal as _process_signal
from .eval_discrete_signal import eval_discrete_signal


def _resolve_signal(expression, grid_ids, signal_arg_index: int):
    """Resolve signal from typed expression via get_arg."""
    sig = expression.get_arg(signal_arg_index)
    return eval_discrete_signal(sig, grid_ids)


def _resolve_scalar(expression, arg_index: int):
    """Resolve arg to a Python float for use in tensor arithmetic (avoids sympy/sympify)."""
    v = expression.get_arg(arg_index)
    if hasattr(v, "numel") and hasattr(v, "item"):
        if v.numel() == 1:
            return float(v.item())
        return float(v.flatten()[0].item())
    if hasattr(v, "item"):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


@rec_grid_eval.register
def eval_layout_cell_translate(expression: sws.LayoutCellTranslate, sketcher: Sketcher,
                               grid=None, grid_ids=None):
    layout_expr = expression.args[0]
    grid, grid_ids = rec_grid_eval(layout_expr, sketcher, grid, grid_ids)
    signal = _resolve_signal(expression, grid_ids, 1)
    mode_double = str(expression.args[4]) == "double" if len(expression.args) > 4 else False
    t_x = _resolve_scalar(expression, 2)
    t_y = _resolve_scalar(expression, 3)
    processed = _process_signal(signal.float().unsqueeze(-1), mode_double)
    offset = processed.repeat(1, 2)
    offset[..., 0] = offset[..., 0] * t_x
    offset[..., 1] = offset[..., 1] * t_y
    grid = grid + offset.to(grid.device).to(grid.dtype)
    return grid, grid_ids


@rec_grid_eval.register
def eval_layout_cell_rotate(expression: sws.LayoutCellRotate, sketcher: Sketcher,
                            grid=None, grid_ids=None):
    layout_expr = expression.args[0]
    grid, grid_ids = rec_grid_eval(layout_expr, sketcher, grid, grid_ids)
    signal = _resolve_signal(expression, grid_ids, 1)
    mode_double = str(expression.args[3]) == "double" if len(expression.args) > 3 else False
    rotation = _resolve_scalar(expression, 2)
    processed = _process_signal(signal.float().unsqueeze(-1), mode_double)
    theta = processed[..., 0] * rotation
    c, s = th.cos(theta), th.sin(theta)
    new_x = c * grid[..., 0] - s * grid[..., 1]
    new_y = s * grid[..., 0] + c * grid[..., 1]
    grid = th.stack([new_x, new_y], dim=-1)
    return grid, grid_ids


@rec_grid_eval.register
def eval_layout_cell_scale(expression: sws.LayoutCellScale, sketcher: Sketcher,
                           grid=None, grid_ids=None):
    layout_expr = expression.args[0]
    grid, grid_ids = rec_grid_eval(layout_expr, sketcher, grid, grid_ids)
    signal = _resolve_signal(expression, grid_ids, 1)
    processed = _process_signal(signal.float().unsqueeze(-1), False)
    scale = _resolve_scalar(expression, 2)
    scale_factor = 1.0 - processed * scale
    # Avoid division by zero: clamp scale_factor away from zero
    scale_factor = scale_factor.to(grid.device).to(grid.dtype)
    scale_factor = th.clamp(scale_factor, min=1e-6)
    grid = grid / scale_factor
    return grid, grid_ids


@rec_grid_eval.register
def eval_layout_cell_reflect(expression: sws.LayoutCellReflect, sketcher: Sketcher,
                             grid=None, grid_ids=None):
    layout_expr = expression.args[0]
    grid, grid_ids = rec_grid_eval(layout_expr, sketcher, grid, grid_ids)
    signal = _resolve_signal(expression, grid_ids, 1)
    processed = _process_signal(signal.float().unsqueeze(-1), False)
    mask = (processed[..., 0] > 0.5)
    reflect = str(expression.args[2])
    ref_grid = grid.clone()
    if reflect == "x":
        ref_grid[..., 0] = -ref_grid[..., 0]
    elif reflect == "y":
        ref_grid[..., 1] = -ref_grid[..., 1]
    else:
        ref_grid = -ref_grid
    grid = th.where(mask.unsqueeze(-1), grid, ref_grid)
    return grid, grid_ids

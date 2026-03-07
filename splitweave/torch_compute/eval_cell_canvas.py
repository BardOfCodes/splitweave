"""
Cell canvas effect evaluation for splitweave.

Uses singledispatch on cell canvas effect types (CellRecolor, CellOutline,
CellOpacity, CellShadow) to apply per-cell visual effects to a canvas.

API
---
rec_eval_cell_canvas_effect(expr, canvas, grid_ids, sketcher)
    Apply a cell canvas effect. Returns modified canvas [N, 4] RGBA.

Helpers (used by evaluate.py for ApplyCell* legacy path):
    process_signal, apply_recolor, apply_cell_outline, apply_cell_opacity
"""
import sys
import numpy as np
import torch as th

from geolipi.torch_compute import Sketcher
from geolipi.torch_compute.colorspace_functions import rgb2hsl_torch, hsl2rgb_torch

import splitweave.symbolic as sws
from .eval_discrete_signal import eval_discrete_signal

if sys.version_info >= (3, 11):
    from functools import singledispatch
else:
    from geolipi.torch_compute.patched_functools import singledispatch


# ============================================================
# Helpers (exported for evaluate.py ApplyCell* handlers)
# ============================================================

def process_signal(signal: th.Tensor, mode_double: bool) -> th.Tensor:
    """Normalize signal to [0, 1] and optionally center to [-1, 1] for double mode."""
    signal = signal.float()
    max_val = th.max(th.abs(signal))
    min_val = th.min(th.abs(signal))
    denom = max_val - min_val
    if denom > 0:
        signal = (signal - min_val) / denom
    if mode_double:
        signal = (signal - 0.5) * 2
    if len(signal.shape) == 1:
        signal = signal.unsqueeze(-1)
    return signal


def apply_recolor(recolor_type: int, recolor_seed: int, mode_double: bool,
                  signal: th.Tensor, canvas: th.Tensor, grid_ids: th.Tensor,
                  major_hues: float = 0.0) -> th.Tensor:
    """Apply hue-shift recoloring to canvas based on the discrete signal."""
    st0 = np.random.get_state()
    np.random.seed(recolor_seed)

    if recolor_type == 0:  # smooth
        hue_value = np.random.uniform(0.33, 0.66)
        hue_value = np.random.choice([-1, 1]) * hue_value
        processed_signal = process_signal(signal, mode_double)
        hue_shift_val = processed_signal * hue_value - major_hues
    else:  # discrete
        processed_signal = signal.long()
        unique_ids, inverse_indices = processed_signal.unique(dim=0, return_inverse=True)
        num_unique_ids = unique_ids.size(0)
        hue_shift_val = np.random.uniform(0, 1, size=(num_unique_ids,))
        hue_shift_val = th.from_numpy(hue_shift_val).to(grid_ids.device).float()
        hue_shift_val = hue_shift_val[inverse_indices].unsqueeze(-1) - major_hues

    np.random.set_state(st0)

    alpha = canvas[..., -1:].clone()
    hsl = rgb2hsl_torch(canvas[..., :-1])
    hsl[..., 0:1] = (hsl[..., 0:1] + hue_shift_val) % 1
    new_canvas = hsl2rgb_torch(hsl)
    new_canvas = th.cat([new_canvas, alpha], dim=-1)
    return new_canvas


def apply_cell_outline(canvas: th.Tensor, grid_ids: th.Tensor, signal: th.Tensor,
                       outline_color, outline_thickness: float, resolution: int) -> th.Tensor:
    """Apply per-cell outline: alpha dilation + blend."""
    processed = process_signal(signal.float().unsqueeze(-1), False)
    mask = (processed[..., 0] > 0.5).float().unsqueeze(-1)
    n = canvas.shape[0]
    side = int(round(n ** 0.5))
    if side * side != n:
        return canvas
    alpha = canvas[..., -1:].reshape(side, side, 1)
    kernel_size = max(3, int(outline_thickness * side * 2) | 1)
    pad = kernel_size // 2
    alpha_pad = th.nn.functional.pad(
        alpha.permute(2, 0, 1).unsqueeze(0), (pad, pad, pad, pad), mode="replicate"
    )
    dilated = th.nn.functional.max_pool2d(alpha_pad, kernel_size, stride=1).squeeze(0).permute(1, 2, 0)
    outline_alpha = (dilated.reshape(n, 1) - alpha.reshape(n, 1)).clamp(0, 1) * mask
    if isinstance(outline_color, th.Tensor):
        color = outline_color.to(canvas.device).to(canvas.dtype)
    else:
        color = th.tensor(outline_color, device=canvas.device, dtype=canvas.dtype)
    if color.dim() == 1:
        color = color.unsqueeze(0).expand(n, -1)
    new_canvas = canvas.clone()
    new_canvas[..., :-1] = (1 - outline_alpha) * canvas[..., :-1] + outline_alpha * color[..., :-1]
    new_canvas[..., -1:] = th.maximum(canvas[..., -1:], outline_alpha * color[..., -1:])
    return new_canvas


def apply_cell_opacity(canvas: th.Tensor, grid_ids: th.Tensor, signal: th.Tensor,
                       opacity: float) -> th.Tensor:
    """Apply per-cell opacity modulation."""
    processed = process_signal(signal.float().unsqueeze(-1), False)
    alpha = canvas[..., -1:] * (1.0 - processed * opacity)
    return th.cat([canvas[..., :-1], alpha], dim=-1)


def _resolve_signal(expr, grid_ids: th.Tensor):
    """Resolve signal from expr via get_arg; typed signal expr -> eval_discrete_signal."""
    if len(expr.args) <= 2:
        return None
    signal_val = expr.get_arg(2)
    return eval_discrete_signal(signal_val, grid_ids)


def _resolve_param(expr, arg_index: int, default=None):
    """Resolve a param from expr via get_arg."""
    if arg_index >= len(expr.args):
        return default
    v = expr.get_arg(arg_index)
    if isinstance(v, th.Tensor) and v.numel() == 1:
        return v.item()
    if v is None:
        return default
    return v


# ============================================================
# Singledispatch for cell canvas effects
# ============================================================

@singledispatch
def rec_eval_cell_canvas_effect(expr, canvas: th.Tensor, grid_ids: th.Tensor,
                                sketcher: Sketcher) -> th.Tensor:
    """Apply a cell canvas effect. Dispatches on expr type."""
    raise TypeError(
        f"rec_eval_cell_canvas_effect: unsupported type {type(expr).__name__}. "
        "Expected CellRecolor, CellOutline, CellOpacity, or CellShadow."
    )


@rec_eval_cell_canvas_effect.register(sws.CellRecolor)
def _eval_cell_recolor(expr: sws.CellRecolor, canvas: th.Tensor, grid_ids: th.Tensor,
                       sketcher: Sketcher) -> th.Tensor:
    signal = _resolve_signal(expr, grid_ids)
    recolor_type_str = str(_resolve_param(expr, 3, "smooth"))
    recolor_seed = int(_resolve_param(expr, 4, 0))
    mode_str = str(_resolve_param(expr, 5, "single"))
    recolor_type = 0 if recolor_type_str == "smooth" else 1
    mode_double = mode_str == "double"
    return apply_recolor(recolor_type, recolor_seed, mode_double, signal, canvas, grid_ids)


@rec_eval_cell_canvas_effect.register(sws.CellOutline)
def _eval_cell_outline(expr: sws.CellOutline, canvas: th.Tensor, grid_ids: th.Tensor,
                       sketcher: Sketcher) -> th.Tensor:
    signal = _resolve_signal(expr, grid_ids)
    outline_color = _resolve_param(expr, 3)
    if isinstance(outline_color, th.Tensor):
        outline_color = outline_color.to(canvas.device)
    else:
        outline_color = th.tensor(outline_color, device=canvas.device, dtype=canvas.dtype)
    thickness = float(_resolve_param(expr, 4, 0.02))
    return apply_cell_outline(canvas, grid_ids, signal, outline_color, thickness, sketcher.resolution)


@rec_eval_cell_canvas_effect.register(sws.CellOpacity)
def _eval_cell_opacity(expr: sws.CellOpacity, canvas: th.Tensor, grid_ids: th.Tensor,
                       sketcher: Sketcher) -> th.Tensor:
    signal = _resolve_signal(expr, grid_ids)
    opacity = float(_resolve_param(expr, 3, 0.5))
    return apply_cell_opacity(canvas, grid_ids, signal, opacity)


@rec_eval_cell_canvas_effect.register(sws.CellShadow)
def _eval_cell_shadow(expr: sws.CellShadow, canvas: th.Tensor, grid_ids: th.Tensor,
                      sketcher: Sketcher) -> th.Tensor:
    """CellShadow: drop shadow - not fully implemented, pass through for now."""
    signal = _resolve_signal(expr, grid_ids)
    # TODO: implement shadow (offset alpha, blur, composite)
    return canvas

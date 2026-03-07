"""
Deformation dispatch handlers for the splitweave grid evaluator.

Registers singledispatch handlers for all DeformGrid subclasses
on the rec_grid_eval dispatcher defined in evaluate.py.
"""
import math
import torch as th

from geolipi.torch_compute import Sketcher
import splitweave.symbolic as sws

from .eval_grid import rec_grid_eval, _parse_sws_params


# ============================================================
# Deformations
# ============================================================

@rec_grid_eval.register
def eval_radial_deform(expression: sws.RadialDeform, sketcher: Sketcher,
                       grid=None, grid_ids=None):
    """Radial zoom/scale deformation based on distance from center."""
    grid, grid_ids = rec_grid_eval(expression.args[0], sketcher, grid, grid_ids)
    params = expression.args[1:]
    params = _parse_sws_params(expression, params)
    # params: center, signal_mode, dist_rate, [sin_k, phase_shift, sigmoid_rate, sigmoid_spread]
    center = params[0]
    signal_mode = str(params[1])
    dist_rate = float(params[2])

    # Compute radial distance from center
    center_t = th.tensor(center, device=grid.device, dtype=grid.dtype) if not isinstance(center, th.Tensor) else center.to(device=grid.device, dtype=grid.dtype)
    radial_grid = th.sqrt((grid[..., 0] - center_t[0])**2 + (grid[..., 1] - center_t[1])**2)
    radial_grid = radial_grid / (radial_grid.max() + 1e-8)
    radial_grid = radial_grid.unsqueeze(-1)

    if signal_mode == "linear":
        deform_amt = 1.0 + radial_grid * dist_rate
    elif signal_mode == "siny":
        sin_k = float(params[3]) if len(params) > 3 else 6.0
        phase_shift = float(params[4]) if len(params) > 4 else 0.0
        deform_amt = th.sin(radial_grid * sin_k * math.pi + phase_shift)
        deform_amt = 1.0 + ((1.0 + deform_amt) / 2.0) * dist_rate / 4.0
    elif signal_mode == "sigmoid":
        sigmoid_rate = float(params[5]) if len(params) > 5 else -20.0
        sigmoid_spread = float(params[6]) if len(params) > 6 else 0.35
        deform_amt = 1.0 / (1.0 + th.exp(sigmoid_rate * (radial_grid - sigmoid_spread)))
        deform_amt = 1.0 + deform_amt * dist_rate * 0.5
    else:
        deform_amt = 1.0 + radial_grid * dist_rate

    grid = grid * deform_amt
    return grid, grid_ids


@rec_grid_eval.register
def eval_perlin_deform(expression: sws.PerlinDeform, sketcher: Sketcher,
                       grid=None, grid_ids=None):
    """Perlin noise offset deformation."""
    grid, grid_ids = rec_grid_eval(expression.args[0], sketcher, grid, grid_ids)
    params = expression.args[1:]
    params = _parse_sws_params(expression, params)
    # params: resolution_seq, seed, dist_mode, dist_rate
    resolution_seq = params[0]
    seed = int(params[1])
    dist_mode = str(params[2])
    dist_rate = float(params[3])

    if isinstance(resolution_seq, th.Tensor):
        resolution_seq = resolution_seq.tolist()
    elif isinstance(resolution_seq, (tuple, list)):
        resolution_seq = [int(x) if isinstance(x, (int, float)) else int(x) for x in resolution_seq]

    th.manual_seed(seed)
    # Sum multi-octave perlin — tensorize leaves individually so that
    # .tensor() never hits a bare sympy Add node.
    dev = str(sketcher.device)
    noise_expr = sws.PerlinNoise((resolution_seq[0],)).tensor(device=dev)
    for rate in resolution_seq[1:]:
        noise_expr = noise_expr + sws.PerlinNoise((rate,)).tensor(device=dev)
    n_grid, _ = rec_grid_eval(noise_expr, sketcher, grid, None)
    n_grid = (n_grid - th.min(n_grid)) / (th.max(n_grid) - th.min(n_grid) + 1e-8)
    n_grid = 2.0 * n_grid - 1.0
    noise_grid = n_grid * dist_rate

    zeros = th.zeros_like(noise_grid)
    offset = th.cat([zeros, zeros], dim=-1)

    if dist_mode == "x":
        offset[:, 0] = noise_grid[:, 0] * 2.0
    elif dist_mode == "y":
        offset[:, 1] = noise_grid[:, 0] * 2.0
    else:  # "xy"
        noise_expr2 = sws.PerlinNoise((resolution_seq[0],)).tensor(device=dev)
        for rate in resolution_seq[1:]:
            noise_expr2 = noise_expr2 + sws.PerlinNoise((rate,)).tensor(device=dev)
        n_grid_2, _ = rec_grid_eval(noise_expr2, sketcher, grid, None)
        n_grid_2 = (n_grid_2 - th.min(n_grid_2)) / (th.max(n_grid_2) - th.min(n_grid_2) + 1e-8)
        n_grid_2 = 2.0 * n_grid_2 - 1.0
        noise_grid_2 = n_grid_2 * dist_rate
        offset[:, 0] = noise_grid[:, 0]
        offset[:, 1] = noise_grid_2[:, 0]

    grid = grid + offset
    return grid, grid_ids


@rec_grid_eval.register
def eval_decay_deform(expression: sws.DecayDeform, sketcher: Sketcher,
                      grid=None, grid_ids=None):
    """Directional decay deformation."""
    grid, grid_ids = rec_grid_eval(expression.args[0], sketcher, grid, grid_ids)
    params = expression.args[1:]
    params = _parse_sws_params(expression, params)
    # params: direction, axis, dist_rate
    direction = params[0]
    axis = str(params[1])
    dist_rate = float(params[2])

    simple_coords = sketcher.get_base_coords().to(grid.device)
    dir_t = th.tensor(direction, device=grid.device, dtype=grid.dtype) if not isinstance(direction, th.Tensor) else direction.to(grid.device)
    dir_t = dir_t / (th.norm(dir_t) + 1e-8)
    dir_t = dir_t.unsqueeze(0).expand(simple_coords.shape[0], -1)

    d_fac = th.sum(simple_coords * -dir_t, dim=-1)
    d_fac = (d_fac - th.min(d_fac)) / (th.max(d_fac) - th.min(d_fac) + 1e-8)
    d_fac = d_fac.unsqueeze(-1) * dist_rate
    decay_factor = 1.0 + d_fac

    ones = th.ones_like(decay_factor)
    if axis == "x":
        scale_vec = th.cat([decay_factor, ones], dim=-1)
    elif axis == "y":
        scale_vec = th.cat([ones, decay_factor], dim=-1)
    else:  # "xy"
        scale_vec = th.cat([decay_factor, decay_factor], dim=-1)

    grid = grid * scale_vec
    return grid, grid_ids


@rec_grid_eval.register
def eval_strip_deform(expression: sws.StripDeform, sketcher: Sketcher,
                      grid=None, grid_ids=None):
    """Sinusoidal strip deformation."""
    grid, grid_ids = rec_grid_eval(expression.args[0], sketcher, grid, grid_ids)
    params = expression.args[1:]
    params = _parse_sws_params(expression, params)
    # params: angle, axis, sin_k, phase_shift, dist_rate
    angle = float(params[0])
    axis = str(params[1])
    sin_k = float(params[2])
    phase_shift = float(params[3])
    dist_rate = float(params[4])

    grid_vals = sketcher.get_base_coords().to(grid.device)
    angle_t = th.tensor(angle, device=grid.device, dtype=grid.dtype)
    in_vals = grid_vals[:, 0:1] * th.sin(angle_t) + grid_vals[:, 1:2] * th.cos(angle_t)
    shift_amt = th.sin(in_vals * sin_k * math.pi + phase_shift) * dist_rate

    zeros = th.zeros_like(shift_amt)
    if axis == "x":
        offset = th.cat([shift_amt, zeros], dim=-1)
    elif axis == "y":
        offset = th.cat([zeros, shift_amt], dim=-1)
    else:  # "xy"
        offset = th.cat([shift_amt, shift_amt], dim=-1)

    grid = grid + offset
    return grid, grid_ids


@rec_grid_eval.register
def eval_swirl_deform(expression: sws.SwirlDeform, sketcher: Sketcher,
                      grid=None, grid_ids=None):
    """Swirl deformation: rotate grid around center by radially-varying angle."""
    grid, grid_ids = rec_grid_eval(expression.args[0], sketcher, grid, grid_ids)
    params = expression.args[1:]
    params = _parse_sws_params(expression, params)
    # params: center, signal_mode, dist_rate, [sin_k, phase_shift, sigmoid_rate, sigmoid_spread]
    center = params[0]
    signal_mode = str(params[1])
    dist_rate = float(params[2])

    center_t = th.tensor(center, device=grid.device, dtype=grid.dtype) if not isinstance(center, th.Tensor) else center.to(device=grid.device, dtype=grid.dtype)
    radial_grid = th.sqrt((grid[..., 0] - center_t[0])**2 + (grid[..., 1] - center_t[1])**2)
    radial_grid = radial_grid / (radial_grid.max() + 1e-8)
    radial_grid = radial_grid.unsqueeze(-1)

    if signal_mode == "linear":
        deform_amt = radial_grid * dist_rate
    elif signal_mode == "siny":
        sin_k = float(params[3]) if len(params) > 3 else 6.0
        phase_shift = float(params[4]) if len(params) > 4 else 0.0
        deform_amt = th.sin(radial_grid * sin_k * math.pi + phase_shift)
        deform_amt = ((1.0 + deform_amt) / 2.0) * dist_rate / 16.0
    elif signal_mode == "sigmoid":
        sigmoid_rate = float(params[5]) if len(params) > 5 else -20.0
        sigmoid_spread = float(params[6]) if len(params) > 6 else 0.35
        deform_amt = 1.0 / (1.0 + th.exp(sigmoid_rate * (radial_grid - sigmoid_spread)))
        deform_amt = deform_amt * dist_rate * 0.5
    else:
        deform_amt = radial_grid * dist_rate

    angle = deform_amt.squeeze(-1)
    centered = grid - center_t.unsqueeze(0)
    coses = th.cos(angle)
    sines = th.sin(angle)
    rot_x = centered[..., 0] * coses - centered[..., 1] * sines
    rot_y = centered[..., 0] * sines + centered[..., 1] * coses
    grid = th.stack([rot_x, rot_y], dim=-1) + center_t.unsqueeze(0)
    return grid, grid_ids


@rec_grid_eval.register
def eval_no_deform(expression: sws.NoDeform, sketcher: Sketcher,
                   grid=None, grid_ids=None):
    """Identity deformation -- pass through."""
    grid, grid_ids = rec_grid_eval(expression.args[0], sketcher, grid, grid_ids)
    return grid, grid_ids

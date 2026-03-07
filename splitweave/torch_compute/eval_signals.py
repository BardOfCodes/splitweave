"""
Signal dispatch handlers for the splitweave grid evaluator.

Registers singledispatch handlers for all Signal2D subclasses
on the rec_grid_eval dispatcher defined in evaluate.py.
"""
import math
import torch as th

from geolipi.torch_compute import Sketcher
import splitweave.symbolic as sws

from .eval_grid import rec_grid_eval, _parse_sws_params


# ============================================================
# Signals
# ============================================================

@rec_grid_eval.register
def eval_radial_signal(expression: sws.RadialSignal, sketcher: Sketcher,
                       grid=None, grid_ids=None):
    """Evaluate a radial signal to produce a scalar field."""
    params = expression.args
    params = _parse_sws_params(expression, params)
    # params: center, signal_mode, [sin_k, phase_shift, sigmoid_rate, sigmoid_spread]
    center = params[0]
    signal_mode = str(params[1])

    coords = sketcher.get_base_coords()
    center_t = th.tensor(center, device=coords.device, dtype=coords.dtype) if not isinstance(center, th.Tensor) else center
    radial_grid = th.sqrt((coords[..., 0] - center_t[0])**2 + (coords[..., 1] - center_t[1])**2)
    radial_grid = radial_grid / (radial_grid.max() + 1e-8)
    radial_grid = radial_grid.unsqueeze(-1)

    if signal_mode == "linear":
        signal = radial_grid
    elif signal_mode == "siny":
        sin_k = float(params[2]) if len(params) > 2 else 6.0
        phase_shift = float(params[3]) if len(params) > 3 else 0.0
        signal = th.sin(radial_grid * sin_k * math.pi + phase_shift)
    elif signal_mode == "sigmoid":
        sigmoid_rate = float(params[4]) if len(params) > 4 else -20.0
        sigmoid_spread = float(params[5]) if len(params) > 5 else 0.35
        signal = 1.0 / (1.0 + th.exp(sigmoid_rate * (radial_grid - sigmoid_spread)))
    elif signal_mode == "exponential":
        signal = th.exp(radial_grid) - 1.0
    else:
        signal = radial_grid

    # Normalize to [0, 1]
    signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
    return signal, grid_ids


@rec_grid_eval.register
def eval_perlin_signal(expression: sws.PerlinSignal, sketcher: Sketcher,
                       grid=None, grid_ids=None):
    """Evaluate multi-octave Perlin noise signal."""
    params = expression.args
    params = _parse_sws_params(expression, params)
    resolution_seq = params[0]
    seed = int(params[1])

    if isinstance(resolution_seq, th.Tensor):
        resolution_seq = resolution_seq.tolist()

    th.manual_seed(seed)
    dev = str(sketcher.device)
    noise_expr = sws.PerlinNoise((resolution_seq[0],)).tensor(device=dev)
    for rate in resolution_seq[1:]:
        noise_expr = noise_expr + sws.PerlinNoise((rate,)).tensor(device=dev)
    n_grid, _ = rec_grid_eval(noise_expr, sketcher, grid, None)
    n_grid = (n_grid - th.min(n_grid)) / (th.max(n_grid) - th.min(n_grid) + 1e-8)
    return n_grid, grid_ids


@rec_grid_eval.register
def eval_decay_signal(expression: sws.DecaySignal, sketcher: Sketcher,
                      grid=None, grid_ids=None):
    """Evaluate directional decay signal."""
    params = expression.args
    params = _parse_sws_params(expression, params)
    direction = params[0]

    simple_coords = sketcher.get_base_coords()
    dir_t = th.tensor(direction, device=simple_coords.device, dtype=simple_coords.dtype) if not isinstance(direction, th.Tensor) else direction
    dir_t = dir_t / (th.norm(dir_t) + 1e-8)
    dir_t = dir_t.unsqueeze(0).expand(simple_coords.shape[0], -1)

    d_fac = th.sum(simple_coords * -dir_t, dim=-1)
    d_fac = (d_fac - th.min(d_fac)) / (th.max(d_fac) - th.min(d_fac) + 1e-8)
    d_fac = d_fac.unsqueeze(-1)
    return d_fac, grid_ids


@rec_grid_eval.register
def eval_strip_signal(expression: sws.StripSignal, sketcher: Sketcher,
                      grid=None, grid_ids=None):
    """Evaluate sinusoidal strip signal."""
    params = expression.args
    params = _parse_sws_params(expression, params)
    angle = float(params[0])
    sin_k = float(params[1])
    phase_shift = float(params[2])

    grid_vals = sketcher.get_base_coords()
    angle_t = th.tensor(angle, device=grid_vals.device, dtype=grid_vals.dtype)
    in_vals = grid_vals[:, 0:1] * th.sin(angle_t) + grid_vals[:, 1:2] * th.cos(angle_t)
    shift_amt = th.sin(in_vals * sin_k * math.pi + phase_shift)
    shift_amt = (shift_amt - th.min(shift_amt)) / (th.max(shift_amt) - th.min(shift_amt) + 1e-8)
    return shift_amt, grid_ids


@rec_grid_eval.register
def eval_swirl_signal(expression: sws.SwirlSignal, sketcher: Sketcher,
                      grid=None, grid_ids=None):
    """Evaluate swirl signal (same as radial signal)."""
    params = expression.args
    params = _parse_sws_params(expression, params)
    center = params[0]
    signal_mode = str(params[1])

    coords = sketcher.get_base_coords()
    center_t = th.tensor(center, device=coords.device, dtype=coords.dtype) if not isinstance(center, th.Tensor) else center
    radial_grid = th.sqrt((coords[..., 0] - center_t[0])**2 + (coords[..., 1] - center_t[1])**2)
    radial_grid = radial_grid / (radial_grid.max() + 1e-8)
    radial_grid = radial_grid.unsqueeze(-1)

    if signal_mode == "linear":
        signal = radial_grid
    elif signal_mode == "siny":
        sin_k = float(params[2]) if len(params) > 2 else 6.0
        phase_shift = float(params[3]) if len(params) > 3 else 0.0
        signal = th.sin(radial_grid * sin_k * math.pi + phase_shift)
    elif signal_mode == "sigmoid":
        sigmoid_rate = float(params[4]) if len(params) > 4 else -20.0
        sigmoid_spread = float(params[5]) if len(params) > 5 else 0.35
        signal = 1.0 / (1.0 + th.exp(sigmoid_rate * (radial_grid - sigmoid_spread)))
    else:
        signal = radial_grid

    signal = (signal - signal.min()) / (signal.max() - signal.min() + 1e-8)
    return signal, grid_ids


@rec_grid_eval.register
def eval_discrete_signal_grid(expression: sws.DiscreteSignalBase, sketcher: Sketcher,
                              grid=None, grid_ids=None):
    """Discrete per-cell signals are evaluated lazily by eval_discrete_signal(expr, grid_ids).
    When encountered in grid context, pass through (grid, grid_ids)."""
    return grid, grid_ids

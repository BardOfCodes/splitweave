"""
Discrete signal evaluation for splitweave.

Uses singledispatch on signal expression types to compute per-cell
condition tensors from grid_ids.

API
---
eval_discrete_signal(expr, grid_ids)
    Singledispatch on typed signal expressions (CheckerboardSignal, etc.).
"""
import sys
import torch as th
import sympy as sp
import splitweave.symbolic as sws

if sys.version_info >= (3, 11):
    from functools import singledispatch
else:
    from geolipi.torch_compute.patched_functools import singledispatch


def _resolve(expr, index: int, default=None):
    """Resolve expr.get_arg(index) to a scalar (for use in _params)."""
    if index >= len(expr.args):
        return default
    v = expr.get_arg(index)
    if isinstance(v, th.Tensor):
        return v.item() if v.numel() == 1 else v.flatten()[0].item()
    if v is None:
        return default
    try:
        return float(v) if isinstance(v, (int, float, sp.Float, sp.Integer)) else bool(v)
    except (TypeError, ValueError):
        return default


def _params(expr):
    """Extract (k, inverse, group_alternate, apply_sym, double_dip, normalize) from expr. 6-arg spec."""
    n = len(expr.args)
    k = int(_resolve(expr, 0, 2)) if n >= 1 else 2
    inv = bool(_resolve(expr, 1, False)) if n >= 2 else False
    ga = bool(_resolve(expr, 2, False)) if n >= 3 else False
    asym = bool(_resolve(expr, 3, False)) if n >= 4 else False
    dd = bool(_resolve(expr, 4, False)) if n >= 5 else False
    normalize = bool(_resolve(expr, 5, True)) if n >= 6 else True
    return k, inv, ga, asym, dd, normalize


# Mode -> (grid_ids, k) -> raw condition. g is already long, 0-based.
def _raw_count_diag_1(g, k): return g.sum(-1)
def _raw_count_x(g, k): return g[..., 0]
def _raw_count_y(g, k): return g[..., 1]
def _raw_count_diag_2(g, k): return g[..., 1] - g[..., 0]
def _raw_checkerboard(g, k): return g.sum(-1) % k
def _raw_x(g, k): return g[..., 0] % k
def _raw_y(g, k): return g[..., 1] % k
def _raw_xx(g, k): return th.clip(g[..., 0] % (k + 1), 0, k)
def _raw_yy(g, k): return th.clip(g[..., 1] % (k + 1), 0, k)
def _raw_xy(g, k): return th.minimum(g[..., 0] % k, g[..., 1] % k)
def _raw_diag_1(g, k): return th.clip(g.sum(-1) % (k + 1), 0, k)
def _raw_diag_2(g, k): return th.clip((g[..., 1] - g[..., 0]) % (k + 1), 0, k)

def _raw_random(g, k):
    uid, inv = g.float().unique(dim=0, return_inverse=True)
    return (th.rand_like(uid) // (1.0 / k))[inv][..., 0]

def _raw_fully_random(g, k):
    uid, inv = g.unique(dim=0, return_inverse=True)
    return th.rand(uid.shape[0], 1, device=g.device)[inv].squeeze(-1)

_MODE_FN = {
    "count_diag_1": _raw_count_diag_1, "count_x": _raw_count_x, "count_y": _raw_count_y,
    "count_diag_2": _raw_count_diag_2, "checkerboard": _raw_checkerboard,
    "x": _raw_x, "y": _raw_y, "xx": _raw_xx, "yy": _raw_yy, "xy": _raw_xy,
    "diag_1": _raw_diag_1, "diag_2": _raw_diag_2,
    "random": _raw_random, "fully_random": _raw_fully_random,
}

# apply_sym: map condition by max value
_APPLY_SYM_MAP = {
    3: [(3, 1)], 4: [(4, 1), (3, 2)], 5: [(5, 1), (4, 2)], 6: [(6, 1), (5, 2), (4, 3)],
    7: [(7, 1), (6, 2), (5, 3)],
}


def _eval_mode(grid_ids: th.Tensor, mode: str, k: int, inv: bool, ga: bool, asym: bool, dd: bool, normalize: bool=True) -> th.Tensor:
    """Core evaluation: raw condition + postprocess."""
    if normalize:
        g = grid_ids.long() - grid_ids.long().min()
    else:
        g = grid_ids.long()
    fn = _MODE_FN.get(mode, _raw_checkerboard)
    c = fn(g, k)

    if asym:
        for from_v, to_v in _APPLY_SYM_MAP.get(int(c.max().item()), []):
            c = th.where(c == from_v, th.full_like(c, to_v), c)
    if inv and mode != "fully_random":
        c = c.max() - c
    if dd:
        c = th.maximum(c, c.max() - c)
    if ga and mode not in ("xx", "yy"):
        c = th.where(c % 2 == 0, th.zeros_like(c), c)

    # If result is constant, return as-is; do not recurse (would infinite loop)
    return c


# Singledispatch
@singledispatch
def eval_discrete_signal(expr, grid_ids: th.Tensor) -> th.Tensor:
    """Evaluate a discrete signal expression at grid_ids."""
    raise TypeError(f"eval_discrete_signal: unsupported type {type(expr).__name__}")


def eval_discrete_signal_from_params(grid_ids, mode, k, inverse, noise_rate, group_alternate, apply_sym, double_dip):
    """Compute signal from raw params. noise_rate reserved (ignored)."""
    return _eval_mode(grid_ids, mode, k, inverse, group_alternate, apply_sym, double_dip)


@eval_discrete_signal.register(sws.CheckerboardSignal)
def _(expr, grid_ids): return _eval_mode(grid_ids, "checkerboard", *_params(expr))


@eval_discrete_signal.register(sws.XStripeSignal)
def _(expr, grid_ids): return _eval_mode(grid_ids, "x", *_params(expr))


@eval_discrete_signal.register(sws.YStripeSignal)
def _(expr, grid_ids): return _eval_mode(grid_ids, "y", *_params(expr))


@eval_discrete_signal.register(sws.XXStripeSignal)
def _(expr, grid_ids): return _eval_mode(grid_ids, "xx", *_params(expr))


@eval_discrete_signal.register(sws.YYStripeSignal)
def _(expr, grid_ids): return _eval_mode(grid_ids, "yy", *_params(expr))


@eval_discrete_signal.register(sws.XYStripeSignal)
def _(expr, grid_ids): return _eval_mode(grid_ids, "xy", *_params(expr))


@eval_discrete_signal.register(sws.RandomSignal)
def _(expr, grid_ids): return _eval_mode(grid_ids, "random", *_params(expr))


@eval_discrete_signal.register(sws.FullyRandomSignal)
def _(expr, grid_ids): return _eval_mode(grid_ids, "fully_random", *_params(expr))


@eval_discrete_signal.register(sws.DiagonalSignal)
def _(expr, grid_ids):
    axis = str(_resolve(expr, 5, "diag_1")) if len(expr.args) >= 6 else "diag_1"
    return _eval_mode(grid_ids, axis, *_params(expr))


@eval_discrete_signal.register(sws.CountSignal)
def _(expr, grid_ids):
    axis = str(_resolve(expr, 5, "count_x")) if len(expr.args) >= 6 else "count_x"
    return _eval_mode(grid_ids, axis, *_params(expr))

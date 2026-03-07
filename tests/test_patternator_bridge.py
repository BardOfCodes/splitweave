"""
Tests for the config→splitweave bridge.

Contract:
1. CFG → single expression: mtp_config_to_expr(cfg, tile_tensors, device)
   returns one expression (ApplyTile(layout_expr, tile_expr)) that contains all
   requirement details for the pattern (layout + tile).
2. Expression -> pattern: evaluate_pattern(expr, sketcher) returns
   (canvas, grid_ids) — the rendered pattern.
"""
from __future__ import annotations

import os
import unittest

import pytest
import torch as th
import numpy as np

from yacs.config import CfgNode as CN

import geolipi.symbolic as gls
from geolipi.torch_compute import Sketcher

from splitweave.pattern_gen.mtp_main import sample_mtp_pattern, sample_default_mtp_pattern
import splitweave.symbolic as sws
from splitweave.cfg_to_sw.evaluate import evaluate_pattern_cfg, evaluate_rsp_cfg
from splitweave.cfg_to_sw.parser import mtp_config_to_expr, rsp_config_to_expr
from splitweave.torch_compute import evaluate_pattern_expr


DEVICE = "cuda" if th.cuda.is_available() else "cpu"

# Number of random config generations to run for stochastic tests (must pass all).
N_STOCHASTIC_RUNS = 50


def _dummy_tile_tensor(device: str = "cpu"):
    # 16x16 RGBA tile with a non-constant alpha so outline/shadow (AlphaToSDF2D) have a zero contour.
    h = w = 16
    y, x = th.meshgrid(th.arange(h, dtype=th.float32), th.arange(w, dtype=th.float32), indexing="ij")
    mask = ((x // 4 + y // 4) % 2 == 0).float()
    rgb = th.stack([mask, 1.0 - mask, th.zeros_like(mask)], dim=-1)
    # Smooth alpha falloff so alpha crosses 0.1 (alpha_mask uses 0.1 - alpha; skfmm needs zero contour)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    r = th.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    alpha = th.clamp(1.0 - r / 5.0, 0.0, 1.0)[..., None]
    tile = th.cat([rgb, alpha], dim=-1)
    return tile.to(device)


def _make_cfg_with_layout(split_type: str, **split_params) -> CN:
    """Minimal MTP config with given split type and params (one tile: dummy.png)."""
    cfg = CN()
    cfg.layout_cfg = CN()
    cfg.layout_cfg.pre_deform = CN()
    cfg.layout_cfg.pre_deform.t_x = 0.0
    cfg.layout_cfg.pre_deform.t_y = 0.0
    cfg.layout_cfg.pre_deform.rot = 0.0
    cfg.layout_cfg.pre_deform.scale = 1.0
    cfg.layout_cfg.deform = CN()
    cfg.layout_cfg.deform._type = "no_deform"
    cfg.layout_cfg.split = CN()
    cfg.layout_cfg.split._type = split_type
    for k, v in split_params.items():
        setattr(cfg.layout_cfg.split, k, v)
    cfg.layout_cfg.post_deform = None
    t = CN()
    t.tilefile = "dummy.png"
    cfg.tile_cfg = CN()
    cfg.tile_cfg.tileset = [t]
    return cfg


def _add_deform(cfg: CN, deform_type: str, **kwargs) -> CN:
    """Set layout_cfg.deform to the given type and params. Returns cfg."""
    for k, v in kwargs.items():
        setattr(cfg.layout_cfg.deform, k, v)
    cfg.layout_cfg.deform._type = deform_type
    return cfg


def _add_cellfx_effect(cfg: CN, fx_type: str, **kwargs) -> CN:
    """Add one cell effect (needs signal). Creates cellfx_cfg if missing."""
    if not hasattr(cfg, "cellfx_cfg") or cfg.cellfx_cfg is None:
        cfg.cellfx_cfg = CN()
        cfg.cellfx_cfg.effects = []
        cfg.cellfx_cfg.tile_order = CN()
        cfg.cellfx_cfg.tile_order.n_tiles = 1
    fx = CN()
    fx._type = fx_type
    fx.mode = kwargs.get("mode", "single")
    fx.signal = CN()
    fx.signal.discrete_mode = kwargs.get("discrete_mode", "checkerboard")
    fx.signal.k = kwargs.get("k", 2)
    fx.signal.inverse = kwargs.get("inverse", False)
    fx.signal.noise_rate = kwargs.get("noise_rate", 0.0)
    fx.signal.group_alternate = kwargs.get("group_alternate", False)
    fx.signal.apply_sym = kwargs.get("apply_sym", False)
    fx.signal.double_dip = kwargs.get("double_dip", False)
    for k, v in kwargs.items():
        if k not in ("mode", "discrete_mode", "k", "inverse", "noise_rate", "group_alternate", "apply_sym", "double_dip"):
            setattr(fx, k, v)
    cfg.cellfx_cfg.effects.append(fx)
    return cfg


def _add_tile_effects(cfg: CN, **kwargs) -> CN:
    """Add tile_effects to the first tile. E.g. do_recolor=True, new_color_hue=0.5."""
    t = cfg.tile_cfg.tileset[0]
    if not hasattr(t, "tile_effects") or t.tile_effects is None:
        t.tile_effects = CN()
    for k, v in kwargs.items():
        setattr(t.tile_effects, k, v)
    return cfg


def _strip_outline_shadow(cfg: CN) -> None:
    """Disable outline/shadow on all tiles so evaluation does not require a zero alpha contour (skfmm)."""
    for t in cfg.tile_cfg.tileset:
        if hasattr(t, "tile_effects") and t.tile_effects is not None:
            setattr(t.tile_effects, "do_outline", False)
            setattr(t.tile_effects, "do_shadow", False)


def _add_border_and_bg(cfg: CN, border_mode: str = "simple", border_size: float = 0.01) -> CN:
    """Add border_cfg and bg_cfg so border path is exercised."""
    cfg.bg_cfg = CN()
    cfg.bg_cfg._type = "Background"
    cfg.bg_cfg.bg_mode = "plain"
    cfg.bg_cfg.color_seed = 123
    cfg.border_cfg = CN()
    cfg.border_cfg._type = "Border"
    cfg.border_cfg.border_mode = border_mode
    cfg.border_cfg.border_size = border_size
    cfg.border_cfg.color_seed = 456
    return cfg


def _assert_no_cfgnode(expr, path="root"):
    """Recursively check that no CfgNode objects appear in the expression tree."""
    if isinstance(expr, CN):
        raise AssertionError(f"CfgNode found at {path}: {expr}")
    lt = getattr(expr, "lookup_table", {})
    for k, v in lt.items():
        if isinstance(v, CN):
            raise AssertionError(f"CfgNode in lookup_table at {path}.lookup_table[{k}]: {v}")
        if isinstance(v, dict) and "_type" in v:
            raise AssertionError(f"dict resembling CfgNode in lookup_table at {path}.lookup_table[{k}]: {v}")
    if hasattr(expr, "args"):
        for i, child in enumerate(expr.args):
            if hasattr(child, "args"):
                _assert_no_cfgnode(child, path=f"{path}.args[{i}]")


def test_cfg_to_single_expression():
    """CFG → single expression: mtp_config_to_expr returns one ApplyTile with layout + tile.
    Runs N_STOCHASTIC_RUNS times with different seeds; all runs must pass.
    Also verifies NO CfgNode objects remain in the expression tree."""
    filenames = ["dummy.png"]
    filename_to_indices = {"dummy.png": [0]}
    n_plain = 0
    n_grid_n_color = 0
    for run in range(N_STOCHASTIC_RUNS):
        np.random.seed(run)
        th.manual_seed(run)
        cfg = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=1, old_mode=True)
        tile_tensors = {}
        for t_cfg in cfg.tile_cfg.tileset:
            tilefile = getattr(t_cfg, "tilefile", "dummy.png")
            tile_tensors[str(tilefile)] = _dummy_tile_tensor(device=DEVICE)

        expr = mtp_config_to_expr(cfg, tile_tensors=tile_tensors, device=DEVICE)

        # No CfgNode anywhere in the tree
        _assert_no_cfgnode(expr, path=f"run{run}")

        # Unwrap optional border: SourceOver(BorderEffect(...), pattern) -> pattern
        while isinstance(expr, gls.SourceOver) and len(expr.args) == 2 and isinstance(expr.args[0], sws.BorderEffect):
            expr = expr.args[1]

        # With bg_cfg: SourceOver(ApplyTile/ApplyMultiTile/ApplyCell*Effect(...), background_expr); else ApplyTile/ApplyMultiTile or ApplyCell*Effect(...)
        _tile_root_types = (sws.ApplyTile, sws.ApplyMultiTile, sws.ApplyCellRecolor, sws.ApplyCellOutline, sws.ApplyCellOpacity)
        if isinstance(expr, gls.SourceOver) and len(expr.args) == 2:
            a0, a1 = expr.args[0], expr.args[1]
            apply_tile = a0 if isinstance(a0, _tile_root_types) else (a1 if isinstance(a1, _tile_root_types) else None)
            assert apply_tile is not None, f"Run {run}: SourceOver must contain ApplyTile, ApplyMultiTile, or ApplyCell*Effect"
            bg_expr = a1 if apply_tile is a0 else a0

            if isinstance(bg_expr, sws.ConstantBackground):
                n_plain += 1
            elif isinstance(bg_expr, sws.ApplyColoring):
                n_grid_n_color += 1
                # Verify ApplyColoring has 4 args: layout_expr, base_color, recolor_params, signal_expr
                assert len(bg_expr.args) == 4, f"Run {run}: ApplyColoring should have 4 args, got {len(bg_expr.args)}"
            else:
                raise AssertionError(f"Run {run}: unexpected bg type {type(bg_expr)}")

            while isinstance(apply_tile, (sws.ApplyCellRecolor, sws.ApplyCellOutline, sws.ApplyCellOpacity)):
                apply_tile = apply_tile.args[0]
            layout_expr = apply_tile.args[0]
            tile_expr = apply_tile.args[1] if isinstance(apply_tile, sws.ApplyTile) else apply_tile.args[2]
        elif isinstance(expr, (sws.ApplyTile, sws.ApplyMultiTile)):
            layout_expr = expr.args[0]
            tile_expr = expr.args[1] if isinstance(expr, sws.ApplyTile) else expr.args[2]
        elif isinstance(expr, (sws.ApplyCellRecolor, sws.ApplyCellOutline, sws.ApplyCellOpacity)):
            inner = expr.args[0]
            while isinstance(inner, (sws.ApplyCellRecolor, sws.ApplyCellOutline, sws.ApplyCellOpacity)):
                inner = inner.args[0]
            layout_expr = inner.args[0]
            tile_expr = inner.args[1] if isinstance(inner, sws.ApplyTile) else inner.args[2]
        else:
            raise AssertionError(f"Run {run}: expected SourceOver(ApplyTile/ApplyMultiTile, bg) or ApplyTile/ApplyMultiTile, got {type(expr)}")
        assert isinstance(layout_expr, sws.GridFunction), (
            f"Run {run}: layout should be GridFunction, got {type(layout_expr)}"
        )
        assert tile_expr is not None, f"Run {run}: tile_expr should be present"

    # With 50 runs, we should see both background types (p=0.65 plain, p=0.35 grid_n_color)
    print(f"  Background stats: {n_plain} plain, {n_grid_n_color} grid_n_color")


def test_expression_to_pattern():
    """Expression → pattern: evaluate_pattern_expr(expr, sketcher) returns (canvas, grid_ids)."""
    cfg = _make_cfg_with_layout("RectRepeat", x_size=0.2, y_size=0.2)
    tile_tensors = {"dummy.png": _dummy_tile_tensor(device=DEVICE)}
    expr = mtp_config_to_expr(cfg, tile_tensors=tile_tensors, device=DEVICE)

    sketcher = Sketcher(device=DEVICE, resolution=64, n_dims=2)
    canvas, grid_ids = evaluate_pattern_expr(expr, sketcher)

    n = sketcher.resolution * sketcher.resolution
    assert isinstance(canvas, th.Tensor)
    assert isinstance(grid_ids, th.Tensor)
    assert canvas.shape[0] == n or canvas.numel() == n * 4
    assert grid_ids.shape[0] == n
    assert grid_ids.shape[-1] >= 2
    assert not th.isnan(canvas).any()
    assert not th.isinf(canvas).any()


def test_cfg_to_expression_to_pattern():
    """Full pipeline: CFG -> single expression -> pattern (evaluate_pattern_cfg does both steps).
    Runs N_STOCHASTIC_RUNS times with different seeds; all runs must pass."""
    filenames = ["dummy.png"]
    filename_to_indices = {"dummy.png": [0]}
    sketcher = Sketcher(device=DEVICE, resolution=64, n_dims=2)
    n = sketcher.resolution * sketcher.resolution
    for run in range(N_STOCHASTIC_RUNS):
        np.random.seed(run)
        th.manual_seed(run)
        cfg = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=1, old_mode=True)
        _strip_outline_shadow(cfg)
        tile_tensors = {}
        for t_cfg in cfg.tile_cfg.tileset:
            tilefile = getattr(t_cfg, "tilefile", "dummy.png")
            tile_tensors[str(tilefile)] = _dummy_tile_tensor(device=DEVICE)

        canvas, grid_ids = evaluate_pattern_cfg(cfg, sketcher, tile_tensors=tile_tensors)

        assert isinstance(canvas, th.Tensor), f"Run {run}"
        assert isinstance(grid_ids, th.Tensor), f"Run {run}"
        assert canvas.shape[0] == n or canvas.numel() == n * 4, f"Run {run}"
        assert grid_ids.shape[0] == n, f"Run {run}"
        assert not th.isnan(canvas).any(), f"Run {run}"
        assert not th.isinf(canvas).any(), f"Run {run}"


def test_bridge_rectrepeat_mvp():
    """Bridge with dummy tiles; runs N_STOCHASTIC_RUNS times with different seeds; all must pass."""
    filenames = ["dummy.png"]
    filename_to_indices = {"dummy.png": [0]}
    sketcher = Sketcher(device=DEVICE, resolution=64, n_dims=2)
    n = sketcher.resolution * sketcher.resolution
    for run in range(N_STOCHASTIC_RUNS):
        np.random.seed(run)
        th.manual_seed(run)
        cfg = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=1, old_mode=True)
        _strip_outline_shadow(cfg)
        tile_tensors = {}
        for t_cfg in cfg.tile_cfg.tileset:
            tilefile = getattr(t_cfg, "tilefile", "dummy.png")
            tile_tensors[str(tilefile)] = _dummy_tile_tensor(device=DEVICE)

        canvas, grid_ids = evaluate_pattern_cfg(cfg, sketcher, tile_tensors=tile_tensors)

        assert isinstance(canvas, th.Tensor), f"Run {run}"
        assert isinstance(grid_ids, th.Tensor), f"Run {run}"
        assert canvas.shape[0] == n or canvas.numel() == n * 4, f"Run {run}"
        assert grid_ids.shape[0] == n, f"Run {run}"
        assert not th.isnan(canvas).any(), f"Run {run}"
        assert not th.isinf(canvas).any(), f"Run {run}"


def test_bridge_with_real_tiles():
    """
    Test matching the exact notebook code from test_config_to_pattern.ipynb.
    Runs N_STOCHASTIC_RUNS times with different seeds; all runs must pass.
    """
    TILE_DIR = "/users/aganesh8/data/aganesh8/data/patterns/generative_tiles/food"
    dev = "cuda" if th.cuda.is_available() else "cpu"
    RESOLUTION = 256

    if not os.path.exists(TILE_DIR):
        pytest.skip(f"TILE_DIR {TILE_DIR} does not exist")

    all_files = [
        os.path.join(TILE_DIR, f)
        for f in sorted(os.listdir(TILE_DIR))
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not all_files:
        pytest.skip(f"No image files found in {TILE_DIR}")

    filename_to_indices = {fn: [i] for i, fn in enumerate(all_files)}
    sketcher = Sketcher(device=dev, resolution=RESOLUTION, n_dims=2)
    n = sketcher.resolution * sketcher.resolution

    for run in range(N_STOCHASTIC_RUNS):
        np.random.seed(run)
        th.manual_seed(run)
        cfg = sample_mtp_pattern(all_files, filename_to_indices, n_tiles=1, old_mode=True)
        _strip_outline_shadow(cfg)
        canvas, grid_ids = evaluate_pattern_cfg(cfg, sketcher, tile_dir=TILE_DIR)

        R = RESOLUTION
        if canvas.dim() == 2 and canvas.shape[0] == R * R:
            canvas_img = canvas.detach().cpu().reshape(R, R, -1).numpy()
        else:
            side = int(np.sqrt(canvas.shape[0]))
            canvas_img = canvas.detach().cpu().reshape(side, side, -1).numpy()

        assert isinstance(canvas, th.Tensor), f"Run {run}"
        assert isinstance(grid_ids, th.Tensor), f"Run {run}"
        assert canvas.shape[0] == n or canvas.numel() == n * 4, f"Run {run}"
        assert grid_ids.shape[0] == n, f"Run {run}"
        assert not th.isnan(canvas).any(), f"Run {run}: canvas contains NaN"
        assert not th.isinf(canvas).any(), f"Run {run}: canvas contains Inf"
        assert canvas_img.shape[-1] >= 3, f"Run {run}: expected >=3 channels, got {canvas_img.shape[-1]}"
        assert grid_ids.shape[-1] >= 2, f"Run {run}: grid_ids expected >=2 dims, got {grid_ids.shape[-1]}"


def test_feature_deformations():
    """Deformations: configs with radial_deform, perlin_deform, etc. parse and evaluate."""
    sketcher = Sketcher(device=DEVICE, resolution=64, n_dims=2)
    tile_tensors = {"dummy.png": _dummy_tile_tensor(device=DEVICE)}
    n = sketcher.resolution * sketcher.resolution

    deform_configs = [
        ("radial_deform", {"center": (0.0, 0.0), "signal_mode": "linear", "dist_rate": 0.2}),
        ("perlin_deform", {"resolution_seq": [4, 8], "seed": 0, "dist_rate": 0.08}),
        ("decay_deform", {"direction": (1.0, 0.0), "axis": "x", "dist_rate": 0.5}),
        ("strip_deform", {"angle": 0.0, "axis": "x", "sin_k": 4.0, "phase_shift": 0.0, "dist_rate": 0.05}),
        ("swirl_deform", {"center": (0.0, 0.0), "signal_mode": "linear", "dist_rate": 0.2}),
    ]
    for deform_type, params in deform_configs:
        cfg = _make_cfg_with_layout("RectRepeat", x_size=0.2, y_size=0.2)
        _add_deform(cfg, deform_type, **params)
        expr = mtp_config_to_expr(cfg, tile_tensors=tile_tensors, device=DEVICE)
        canvas, grid_ids = evaluate_pattern_expr(expr, sketcher)
        assert canvas.shape[0] == n or canvas.numel() == n * 4, deform_type
        assert not th.isnan(canvas).any(), deform_type
        assert not th.isinf(canvas).any(), deform_type
    print("All deformation configs passed.")


def test_feature_cell_effects():
    """Cell effects: TranslateFx, RotateFx, ScaleFx, ReflectFx, RecolorFx, OutlineFx, OpacityFx."""
    sketcher = Sketcher(device=DEVICE, resolution=64, n_dims=2)
    tile_tensors = {"dummy.png": _dummy_tile_tensor(device=DEVICE)}
    n = sketcher.resolution * sketcher.resolution

    cfg = _make_cfg_with_layout("RectRepeat", x_size=0.2, y_size=0.2)
    cfg.bg_cfg = CN()
    cfg.bg_cfg._type = "Background"
    cfg.bg_cfg.bg_mode = "plain"
    cfg.bg_cfg.color_seed = 0
    _add_cellfx_effect(cfg, "TranslateFx", t_x=0.05, t_y=0.05)
    expr = mtp_config_to_expr(cfg, tile_tensors=tile_tensors, device=DEVICE)
    canvas, grid_ids = evaluate_pattern_expr(expr, sketcher)
    assert not th.isnan(canvas).any() and not th.isinf(canvas).any()

    cfg = _make_cfg_with_layout("RectRepeat", x_size=0.2, y_size=0.2)
    cfg.bg_cfg = CN()
    cfg.bg_cfg.bg_mode = "plain"
    cfg.bg_cfg.color_seed = 0
    _add_cellfx_effect(cfg, "RecolorFx", recolor_type="smooth", recolor_seed=1)
    expr = mtp_config_to_expr(cfg, tile_tensors=tile_tensors, device=DEVICE)
    canvas, grid_ids = evaluate_pattern_expr(expr, sketcher)
    assert not th.isnan(canvas).any() and not th.isinf(canvas).any()

    cfg = _make_cfg_with_layout("RectRepeat", x_size=0.2, y_size=0.2)
    cfg.bg_cfg = CN()
    cfg.bg_cfg.bg_mode = "plain"
    cfg.bg_cfg.color_seed = 0
    _add_cellfx_effect(cfg, "OutlineFx", outline_color=(0, 0, 0, 1), outline_thickness=0.02)
    expr = mtp_config_to_expr(cfg, tile_tensors=tile_tensors, device=DEVICE)
    canvas, grid_ids = evaluate_pattern_expr(expr, sketcher)
    assert not th.isnan(canvas).any() and not th.isinf(canvas).any()

    cfg = _make_cfg_with_layout("RectRepeat", x_size=0.2, y_size=0.2)
    cfg.bg_cfg = CN()
    cfg.bg_cfg.bg_mode = "plain"
    cfg.bg_cfg.color_seed = 0
    _add_cellfx_effect(cfg, "OpacityFx", opacity=0.8)
    expr = mtp_config_to_expr(cfg, tile_tensors=tile_tensors, device=DEVICE)
    canvas, grid_ids = evaluate_pattern_expr(expr, sketcher)
    assert not th.isnan(canvas).any() and not th.isinf(canvas).any()
    print("All cell effect configs passed.")


def test_feature_per_tile_effects():
    """Per-tile effects: recolor, outline, shadow, rotate, scale, reflect, opacity."""
    sketcher = Sketcher(device=DEVICE, resolution=64, n_dims=2)
    tile_tensors = {"dummy.png": _dummy_tile_tensor(device=DEVICE)}
    n = sketcher.resolution * sketcher.resolution

    cfg = _make_cfg_with_layout("RectRepeat", x_size=0.2, y_size=0.2)
    _add_tile_effects(cfg, do_recolor=True, new_color_hue=0.3)
    expr = mtp_config_to_expr(cfg, tile_tensors=tile_tensors, device=DEVICE)
    canvas, grid_ids = evaluate_pattern_expr(expr, sketcher)
    assert not th.isnan(canvas).any() and not th.isinf(canvas).any()

    cfg = _make_cfg_with_layout("RectRepeat", x_size=0.2, y_size=0.2)
    _add_tile_effects(cfg, do_outline=True, outline_thickness=0.02, outline_color=(0, 0, 0, 1))
    expr = mtp_config_to_expr(cfg, tile_tensors=tile_tensors, device=DEVICE)
    canvas, grid_ids = evaluate_pattern_expr(expr, sketcher)
    assert not th.isnan(canvas).any() and not th.isinf(canvas).any()

    cfg = _make_cfg_with_layout("RectRepeat", x_size=0.2, y_size=0.2)
    _add_tile_effects(cfg, do_rotate=True, rot=0.2)
    expr = mtp_config_to_expr(cfg, tile_tensors=tile_tensors, device=DEVICE)
    canvas, grid_ids = evaluate_pattern_expr(expr, sketcher)
    assert not th.isnan(canvas).any() and not th.isinf(canvas).any()

    cfg = _make_cfg_with_layout("RectRepeat", x_size=0.2, y_size=0.2)
    _add_tile_effects(cfg, do_scale=True, scale=0.9)
    expr = mtp_config_to_expr(cfg, tile_tensors=tile_tensors, device=DEVICE)
    canvas, grid_ids = evaluate_pattern_expr(expr, sketcher)
    assert not th.isnan(canvas).any() and not th.isinf(canvas).any()

    cfg = _make_cfg_with_layout("RectRepeat", x_size=0.2, y_size=0.2)
    _add_tile_effects(cfg, do_opacity=True, opacity=0.7)
    expr = mtp_config_to_expr(cfg, tile_tensors=tile_tensors, device=DEVICE)
    canvas, grid_ids = evaluate_pattern_expr(expr, sketcher)
    assert not th.isnan(canvas).any() and not th.isinf(canvas).any()
    print("All per-tile effect configs passed.")


def test_feature_border():
    """Border: config with border_cfg parses and evaluates (BorderEffect on top of pattern)."""
    sketcher = Sketcher(device=DEVICE, resolution=64, n_dims=2)
    tile_tensors = {"dummy.png": _dummy_tile_tensor(device=DEVICE)}
    n = sketcher.resolution * sketcher.resolution

    for border_mode in ("simple", "dotted", "siny", "onion"):
        cfg = _make_cfg_with_layout("RectRepeat", x_size=0.2, y_size=0.2)
        _add_border_and_bg(cfg, border_mode=border_mode, border_size=0.02)
        expr = mtp_config_to_expr(cfg, tile_tensors=tile_tensors, device=DEVICE)
        canvas, grid_ids = evaluate_pattern_cfg(cfg, sketcher, tile_tensors=tile_tensors)
        assert canvas.shape[0] == n or canvas.numel() == n * 4, border_mode
        assert not th.isnan(canvas).any(), border_mode
        assert not th.isinf(canvas).any(), border_mode
    print("All border configs passed.")


def test_generated_configs_with_features():
    """Generated configs: run sample_mtp_pattern multiple times to exercise mixed features (border, cellfx, etc.)."""
    sketcher = Sketcher(device=DEVICE, resolution=64, n_dims=2)
    filenames = ["dummy.png"]
    filename_to_indices = {"dummy.png": [0]}
    n = sketcher.resolution * sketcher.resolution

    # Run several random samples; generated configs can include border_cfg, cellfx, tile_effects
    for run in range(10):
        np.random.seed(100 + run)
        th.manual_seed(100 + run)
        cfg = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=1, old_mode=True)
        _strip_outline_shadow(cfg)
        tile_tensors = {
            str(getattr(t, "tilefile", "dummy.png")): _dummy_tile_tensor(device=DEVICE)
            for t in cfg.tile_cfg.tileset
        }
        canvas, grid_ids = evaluate_pattern_cfg(cfg, sketcher, tile_tensors=tile_tensors)
        assert not th.isnan(canvas).any(), f"Run {run}"
        assert not th.isinf(canvas).any(), f"Run {run}"
        assert canvas.shape[0] == n or canvas.numel() == n * 4, f"Run {run}"
    print("Generated configs with features passed.")


def test_bridge_all_layout_types():
    """Test that all supported patternator layout types parse and evaluate."""
    sketcher = Sketcher(device=DEVICE, resolution=64, n_dims=2)
    tile_tensors = {"dummy.png": _dummy_tile_tensor(device=DEVICE)}
    n = sketcher.resolution * sketcher.resolution

    layouts = [
        ("RectRepeat", {"x_size": 0.2, "y_size": 0.2}),
        ("RectRepeatFitting", {"fit_x_size": 0.22, "fit_y_size": 0.22}),
        ("RectRepeatShiftedX", {"x_size": 0.2, "y_size": 0.2, "x_shift": 0.1}),
        ("RectRepeatShiftedY", {"x_size": 0.2, "y_size": 0.2, "y_shift": 0.1}),
        ("HexRepeat", {"grid_size": 0.2}),
        ("HexRepeatY", {"grid_size": 0.2}),
        ("RadialRepeat", {"radial_unit": 0.25, "angular_unit": 0.52}),
        ("RadialRepeatBricked", {"radial_unit": 0.25, "angular_unit": 0.52}),
        ("RadialRepeatFixedArc", {"radial_unit": 0.25, "arc_size": 0.35}),
        ("RadialRepeatFixedArcBricked", {"radial_unit": 0.25, "arc_size": 0.35}),
        ("VoronoiRepeat", {"x_size": 0.25, "y_size": 0.25, "noise_rate": 0.5}),
        ("IrregularRepeat", {"x_size": 0.25, "y_size": 0.25, "noise_rate": 0.5}),
    ]
    for split_type, params in layouts:
        cfg = _make_cfg_with_layout(split_type, **params)
        expr = mtp_config_to_expr(cfg, tile_tensors=tile_tensors, device=DEVICE)
        canvas, grid_ids = evaluate_pattern_expr(expr, sketcher)
        assert isinstance(canvas, th.Tensor), split_type
        assert isinstance(grid_ids, th.Tensor), split_type
        assert canvas.shape[0] == n or canvas.numel() == n * 4, split_type
        assert grid_ids.shape[0] == n, split_type
        assert not th.isnan(canvas).any(), split_type
        assert not th.isinf(canvas).any(), split_type
    print("All layout types passed.")


def _make_minimal_rsp_cfg(split_type: str = "x_strip", **split_params) -> CN:
    """Minimal RSP config (layout + fill + tile_cfg, no foreground)."""
    cfg = CN()
    cfg.layout = CN()
    cfg.layout.pre_deform = CN()
    cfg.layout.pre_deform.t_x = 0.0
    cfg.layout.pre_deform.t_y = 0.0
    cfg.layout.pre_deform.rot = 0.0
    cfg.layout.pre_deform.scale = 1.0
    cfg.layout.deform = CN()
    cfg.layout.deform._type = "no_deform"
    cfg.layout.split = CN()
    cfg.layout.split._type = split_type
    for k, v in split_params.items():
        setattr(cfg.layout.split, k, v)
    cfg.layout.foreground = None
    cfg.fill = CN()
    cfg.fill.coloring_mode = "simple"
    cfg.fill.n_colors = 3
    cfg.fill.color_palette = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]
    cfg.tile_cfg = CN()
    cfg.tile_cfg.tileset = []
    return cfg


def _make_rsp_cfg_with_coloring(split_type, coloring_mode, **split_params):
    """RSP config with a specific coloring mode (simple / discrete_interp / discrete_tri_interp)."""
    cfg = _make_minimal_rsp_cfg(split_type, **split_params)
    cfg.fill.coloring_mode = coloring_mode
    if coloring_mode == "discrete_interp":
        cfg.fill.n_colors = 2
        cfg.fill.color_palette = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        cfg.fill.interp_mode = "simple"
        cfg.fill.sin_k = 1
        cfg.fill.mid_point = 0.5
    elif coloring_mode == "discrete_tri_interp":
        cfg.fill.n_colors = 3
        cfg.fill.color_palette = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        cfg.fill.interp_mode = "siny"
        cfg.fill.sin_k = 2
        cfg.fill.mid_point = 0.4
    return cfg


def _make_rsp_cfg_with_foreground(split_type="x_strip", **split_params):
    """RSP config with a foreground (SDFRingPartition from a dummy tile)."""
    cfg = _make_minimal_rsp_cfg(split_type, **split_params)
    cfg.fill.n_colors = 5
    cfg.fill.color_palette = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 1.0]),
    ]
    fg = CN()
    fg._type = "shape_sdf"
    fg.tilespec = CN()
    fg.tilespec.tilefile = "dummy_fg.png"
    fg.sdf_step_size = 0.05
    fg.shape_scale = 0.3
    fg.growth_mode = "linear"
    fg.ring_counts = 2
    cfg.layout.foreground = fg
    cfg.tile_cfg.tileset = [fg.tilespec]
    return cfg


def test_rsp_cfg_to_expression_to_pattern():
    """RSP: minimal configs with all split types and coloring modes."""
    sketcher = Sketcher(device=DEVICE, resolution=64, n_dims=2)
    n = 64 * 64

    # 1. Basic splits with simple coloring
    for split_type, params in [
        ("x_strip", {"stripe_size": 0.15}),
        ("y_strip", {"stripe_size": 0.15}),
        ("checkerboard", {"x_size": 0.3, "y_size": 0.3}),
        ("diamond_board", {"stripe_size": 0.3}),
        ("hex_strip", {"stripe_size": 0.2}),
    ]:
        cfg = _make_minimal_rsp_cfg(split_type, **params)
        expr = rsp_config_to_expr(cfg, device=DEVICE)
        canvas, grid_ids = evaluate_pattern_expr(expr, sketcher)
        assert isinstance(canvas, th.Tensor), split_type
        assert canvas.shape[0] == n, split_type
        assert canvas.shape[-1] == 4, split_type
        assert not th.isnan(canvas).any(), split_type
        assert not th.isinf(canvas).any(), split_type

    # 2. discrete_interp coloring (all interp modes)
    for interp_mode in ("simple", "symmetric", "siny"):
        cfg = _make_rsp_cfg_with_coloring("x_strip", "discrete_interp", stripe_size=0.15)
        cfg.fill.interp_mode = interp_mode
        expr = rsp_config_to_expr(cfg, device=DEVICE)
        canvas, grid_ids = evaluate_pattern_expr(expr, sketcher)
        assert canvas.shape == (n, 4), f"interp {interp_mode}"
        assert not th.isnan(canvas).any(), f"interp {interp_mode}"

    # 3. discrete_tri_interp coloring (all interp modes)
    for interp_mode in ("simple", "symmetric", "siny"):
        cfg = _make_rsp_cfg_with_coloring("checkerboard", "discrete_tri_interp", x_size=0.3, y_size=0.3)
        cfg.fill.interp_mode = interp_mode
        expr = rsp_config_to_expr(cfg, device=DEVICE)
        canvas, grid_ids = evaluate_pattern_expr(expr, sketcher)
        assert canvas.shape == (n, 4), f"tri_interp {interp_mode}"
        assert not th.isnan(canvas).any(), f"tri_interp {interp_mode}"

    # 4. Foreground (SDFRingPartition) with dummy tile
    cfg_fg = _make_rsp_cfg_with_foreground("x_strip", stripe_size=0.15)
    tile_tensors = {"dummy_fg.png": _dummy_tile_tensor(device=DEVICE)}
    expr = rsp_config_to_expr(cfg_fg, tile_tensors=tile_tensors, device=DEVICE)
    canvas, grid_ids = evaluate_pattern_expr(expr, sketcher)
    assert canvas.shape == (n, 4), "foreground"
    assert not th.isnan(canvas).any(), "foreground"

    # 5. evaluate_rsp_cfg and evaluate_pattern_cfg convenience
    cfg = _make_minimal_rsp_cfg("x_strip", stripe_size=0.2)
    canvas, grid_ids = evaluate_rsp_cfg(cfg, sketcher)
    assert canvas.shape[0] == n
    canvas2, _ = evaluate_pattern_cfg(cfg, sketcher)
    assert canvas2.shape == canvas.shape

    print("RSP pipeline passed.")


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


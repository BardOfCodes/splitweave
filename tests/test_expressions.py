"""
Splitweave expression evaluation tests.

Derived from test.ipynb — validates that every expression category
evaluates correctly via the recursive grid parser.

Run:
    conda activate sf
    export PYTHONPATH="..."
    python -m splitweave.tests.test_expressions
"""
import sys
import traceback
import sympy as sp
import torch as th

import splitweave.symbolic as sws
import geolipi.symbolic as gls
from geolipi.torch_compute import Sketcher
from splitweave.torch_compute import grid_eval, rec_grid_eval

# Robust CUDA detection — some nodes have drivers but broken runtime
DEVICE = "cpu"
try:
    if th.cuda.is_available():
        th.zeros(1, device="cuda")  # actually test CUDA
        DEVICE = "cuda"
except Exception:
    DEVICE = "cpu"

RESOLUTION = 256
R = RESOLUTION

sketcher = Sketcher(device=DEVICE, resolution=RESOLUTION, n_dims=2)
print(f"Sketcher ready  |  device={DEVICE}  |  resolution={RESOLUTION}")

# ──────────────────────────────────────────────────────────────
# Helpers — .tensor() on correct device
# ──────────────────────────────────────────────────────────────

def T(expr):
    """Shorthand: materialise symbolic expression on the right device."""
    return expr.tensor(device=DEVICE)


# ──────────────────────────────────────────────────────────────
# Test infrastructure
# ──────────────────────────────────────────────────────────────
PASS = 0
FAIL = 0
ERRORS = []


def check(name, expr, expect_gid=True, min_gid_channels=2):
    """Evaluate an expression and verify basic shape invariants."""
    global PASS, FAIL
    try:
        g, gid = grid_eval(expr, sketcher)

        # grid should be (N, 2) where N = R*R
        assert g.shape == (R * R, 2), \
            f"grid shape {g.shape}, expected ({R*R}, 2)"
        assert not th.isnan(g).any(), "grid contains NaN"
        assert not th.isinf(g).any(), "grid contains Inf"

        if expect_gid:
            assert gid is not None, "expected grid_ids but got None"
            assert gid.shape[0] == R * R, \
                f"grid_ids dim-0 {gid.shape[0]}, expected {R*R}"
            assert gid.shape[-1] >= min_gid_channels, \
                f"grid_ids last dim {gid.shape[-1]}, expected >= {min_gid_channels}"

        PASS += 1
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL += 1
        ERRORS.append((name, e))
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()


def check_scalar(name, expr):
    """Evaluate a signal/noise expression that returns a scalar field."""
    global PASS, FAIL
    try:
        g, gid = grid_eval(expr, sketcher)

        # scalar fields: shape should be (N, 1) or (N,)
        assert g.shape[0] == R * R, \
            f"signal dim-0 {g.shape[0]}, expected {R*R}"
        assert not th.isnan(g).any(), "signal contains NaN"
        assert not th.isinf(g).any(), "signal contains Inf"

        PASS += 1
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL += 1
        ERRORS.append((name, e))
        print(f"  FAIL  {name}: {e}")
        traceback.print_exc()


# ──────────────────────────────────────────────────────────────
# 1. Base Grids
# ──────────────────────────────────────────────────────────────
def test_base_grids():
    print("\n=== 1. Base Grids ===")
    check("CartesianGrid", T(sws.CartesianGrid()), expect_gid=False)
    check("PolarGrid", T(sws.PolarGrid()), expect_gid=False)


# ──────────────────────────────────────────────────────────────
# 2. Rectangular Partitions
# ──────────────────────────────────────────────────────────────
def test_rect_partitions():
    print("\n=== 2. Rectangular Partitions ===")
    exprs = {
        'RectRepeat':   T(sws.RectRepeat(sws.CartesianGrid(), (0.25, 0.25))),
        'RectShiftedX': T(sws.RectRepeatShiftedX(sws.CartesianGrid(), (0.25, 0.25), (0.5,))),
        'RectShiftedY': T(sws.RectRepeatShiftedY(sws.CartesianGrid(), (0.25, 0.25), (0.5,))),
        'RectInner':    T(sws.RectRepeatInner(sws.CartesianGrid(), (0.25, 0.25))),
        'RectFitting':  T(sws.RectRepeatFitting(sws.CartesianGrid(), (0.25, 0.25))),
    }
    for name, expr in exprs.items():
        check(name, expr)


# ──────────────────────────────────────────────────────────────
# 3. Triangular & Diamond Partitions
# ──────────────────────────────────────────────────────────────
def test_tri_diamond():
    print("\n=== 3. Triangular & Diamond ===")
    check("Triangular", T(sws.TriangularRepeat(sws.CartesianGrid(), (0.25,))))
    check("Diamond", T(sws.DiamondRepeat(sws.CartesianGrid(), (0.25,))))


# ──────────────────────────────────────────────────────────────
# 4. Hexagonal Partitions
# ──────────────────────────────────────────────────────────────
def test_hex():
    print("\n=== 4. Hexagonal ===")
    check("HexRepeat", T(sws.HexRepeat(sws.CartesianGrid(), (0.25,))))
    check("HexRepeatX", T(sws.HexRepeatX(sws.CartesianGrid(), (0.25,))))
    check("HexRepeatY", T(sws.HexRepeatY(sws.CartesianGrid(), (0.25,))))


# ──────────────────────────────────────────────────────────────
# 5. Radial / Polar Partitions
# ──────────────────────────────────────────────────────────────
def test_radial():
    print("\n=== 5. Radial / Polar ===")
    polar_base = sws.CartToPolar(sws.CartesianGrid())
    exprs = {
        'RadialAngular':  T(sws.RadialRepeatAngular(polar_base, (0.25,))),
        'RadialCentered': T(sws.RadialRepeatCentered(polar_base, (0.3, 0.3))),
        'RadialBricked':  T(sws.RadialRepeatBricked(polar_base, (0.25, 0.25), (0.1,))),
        'RadialFixedArc': T(sws.RadialRepeatFixedArc(polar_base, (0.25, 0.4), (0.1,))),
    }
    for name, expr in exprs.items():
        check(name, expr)


# ──────────────────────────────────────────────────────────────
# 6. Voronoi & Irregular Partitions
# ──────────────────────────────────────────────────────────────
def test_voronoi():
    print("\n=== 6. Voronoi & Irregular ===")
    check("Voronoi", T(sws.VoronoiRepeat(sws.CartesianGrid(), (0.15, 0.17), (0.3,))))
    check("IrregularRect", T(sws.IrregularRectRepeat(sws.CartesianGrid(), (0.4, 0.4), (0.92,))))
    check("Delaunay", T(sws.DelaunayRepeat(sws.CartesianGrid(), (4.0, 4.0))))


# ──────────────────────────────────────────────────────────────
# 7. Grid Transforms
# ──────────────────────────────────────────────────────────────
def test_transforms():
    print("\n=== 7. Grid Transforms ===")
    base = sws.CartesianGrid()

    # Transform → then RectRepeat to get gid
    check("Translate->Rect",
          T(sws.RectRepeat(sws.CartTranslate(base, (0.2, 0.1)), (0.25, 0.25))))
    check("Scale->Rect",
          T(sws.RectRepeat(sws.CartScale(base, (1.5, 1.2)), (0.25, 0.25))))
    check("Rotate->Rect",
          T(sws.RectRepeat(sws.CartRotate(base, (0.5236,)), (0.25, 0.25))))


# ──────────────────────────────────────────────────────────────
# 8. Scalar Noise Fields
# ──────────────────────────────────────────────────────────────
def test_noise():
    print("\n=== 8. Scalar Noise ===")
    check_scalar("PerlinNoise", T(sws.PerlinNoise((8,))))
    check_scalar("ValueNoise", T(sws.ValueNoise((8,))))

    # Multi-octave arithmetic — tensorize leaves, let evaluator handle Add
    multi = T(sws.PerlinNoise((4,))) + T(sws.PerlinNoise((8,))) + T(sws.PerlinNoise((16,)))
    check_scalar("Perlin_multi_octave", multi)


# ──────────────────────────────────────────────────────────────
# 9. Signal-Driven Transforms
# ──────────────────────────────────────────────────────────────
def test_signal_transforms():
    print("\n=== 9. Signal-Driven Transforms ===")
    # Tensorize sub-parts separately — .tensor() cannot recurse into
    # sympy Add/Mul nodes, so we tensorize each GLFunction leaf and
    # let the evaluator handle the arithmetic composition.
    base = T(sws.RectRepeat(sws.CartesianGrid(), (0.25, 0.25)))
    noise_signal = T(sws.PerlinNoise((4,))) + T(sws.PerlinNoise((8,)))
    noise_signal_expr = noise_signal * gls.Param(th.tensor([0.1]))

    check("TranslateWtSignal", sws.TranslateWtSignal(base, noise_signal_expr))
    check("ScaleWtSignal", sws.ScaleWtSignal(base, noise_signal_expr))


# ──────────────────────────────────────────────────────────────
# 10. Deformations
# ──────────────────────────────────────────────────────────────
def test_deformations():
    print("\n=== 10. Deformations ===")
    base = sws.RectRepeat(sws.CartesianGrid(), (0.25, 0.25))
    center = th.tensor([0.0, 0.0])

    exprs = {
        'Radial(linear)':  T(sws.RadialDeform(base, center, sp.Symbol('linear'), (0.5,))),
        'Radial(siny)':    T(sws.RadialDeform(base, center, sp.Symbol('siny'), (0.5,), (6.0,), (0.0,))),
        'Radial(sigmoid)': T(sws.RadialDeform(base, center, sp.Symbol('sigmoid'), (0.5,),
                                               (0.0,), (0.0,), (-20.0,), (0.35,))),
        'Perlin(xy)':      T(sws.PerlinDeform(base, th.tensor([4, 8, 16]), (42,),
                                               sp.Symbol('xy'), (0.15,))),
        'Decay(x)':        T(sws.DecayDeform(base, th.tensor([1.0, 0.5]),
                                              sp.Symbol('x'), (0.6,))),
        'Strip(xy)':       T(sws.StripDeform(base, (0.785,), sp.Symbol('xy'), (4.0,), (0.0,), (0.08,))),
        'Swirl(linear)':   T(sws.SwirlDeform(base, center, sp.Symbol('linear'), (2.0,))),
    }
    for name, expr in exprs.items():
        check(name, expr)


# ──────────────────────────────────────────────────────────────
# 11. Continuous Signals
# ──────────────────────────────────────────────────────────────
def test_signals():
    print("\n=== 11. Continuous Signals ===")
    center = th.tensor([0.0, 0.0])

    exprs = {
        'Radial(linear)':  T(sws.RadialSignal(center, sp.Symbol('linear'))),
        'Radial(siny)':    T(sws.RadialSignal(center, sp.Symbol('siny'), (6.0,), (0.0,))),
        'Radial(sigmoid)': T(sws.RadialSignal(center, sp.Symbol('sigmoid'),
                                               (0.0,), (0.0,), (-20.0,), (0.35,))),
        'Perlin':          T(sws.PerlinSignal(th.tensor([4, 8, 16]), (42,))),
        'Decay':           T(sws.DecaySignal(th.tensor([1.0, 0.5]), sp.Symbol('xy'))),
        'Strip':           T(sws.StripSignal((0.785,), (4.0,), (0.0,))),
        'Swirl(siny)':     T(sws.SwirlSignal(center, sp.Symbol('siny'), (6.0,), (0.0,))),
    }
    for name, expr in exprs.items():
        check_scalar(name, expr)


# ──────────────────────────────────────────────────────────────
# 12. Edge Masks
# ──────────────────────────────────────────────────────────────
def test_edges():
    print("\n=== 12. Edge Masks ===")
    check("RectRepeatEdge", T(sws.RectRepeatEdge(sws.CartesianGrid(), (0.25, 0.25))))
    check("HexRepeatEdge", T(sws.HexRepeatEdge(sws.CartesianGrid(), (0.25,))))


# ──────────────────────────────────────────────────────────────
# 13. Composed Multi-Stage
# ──────────────────────────────────────────────────────────────
def test_composed():
    print("\n=== 13. Composed Multi-Stage ===")
    center = th.tensor([0.0, 0.0])

    # 1) Swirl → RectRepeat
    expr1 = T(sws.RectRepeat(
        sws.SwirlDeform(sws.CartesianGrid(), center, sp.Symbol('linear'), (1.5,)),
        (0.2, 0.2)
    ))
    check("Swirl->Rect", expr1)

    # 2) Scale → RadialDeform → HexRepeat
    expr2 = T(sws.HexRepeat(
        sws.RadialDeform(
            sws.CartScale(sws.CartesianGrid(), (1.3,)),
            center, sp.Symbol('siny'), (0.4,), (6.0,), (0.0,)
        ),
        (0.2,)
    ))
    check("Scale->Radial->Hex", expr2)

    # 3) StripDeform → Rotate → RectRepeatShiftedX
    expr3 = T(sws.RectRepeatShiftedX(
        sws.CartRotate(
            sws.StripDeform(sws.CartesianGrid(), (0.0,), sp.Symbol('xy'), (6.0,), (0.0,), (0.06,)),
            (0.3,)
        ),
        (0.2, 0.3), (0.5,)
    ))
    check("Strip->Rot->BrickX", expr3)

    # 4) PerlinDeform → Diamond
    expr4 = T(sws.DiamondRepeat(
        sws.PerlinDeform(sws.CartesianGrid(), th.tensor([4, 8]), (7,),
                          sp.Symbol('xy'), (0.12,)),
        (0.2,)
    ))
    check("Perlin->Diamond", expr4)


# ──────────────────────────────────────────────────────────────
# Run all
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_base_grids()
    test_rect_partitions()
    test_tri_diamond()
    test_hex()
    test_radial()
    test_voronoi()
    test_transforms()
    test_noise()
    test_signal_transforms()
    test_deformations()
    test_signals()
    test_edges()
    test_composed()

    print(f"\n{'='*60}")
    print(f"RESULTS:  {PASS} passed,  {FAIL} failed")
    if ERRORS:
        print(f"\nFailed tests:")
        for name, e in ERRORS:
            print(f"  - {name}: {e}")
    print(f"{'='*60}")
    sys.exit(1 if FAIL else 0)

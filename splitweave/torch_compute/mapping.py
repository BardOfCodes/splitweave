"""
Mapping from splitweave symbolic types to torch compute functions.

This maps symbols that use the standard pattern:
    grid_func = sws_to_fn_mapper[expression.__class__]
    result = grid_func(grid, *params)

Deformations, Signals, Cell Effects, Coloring, and Compositing
are evaluated directly in their singledispatch handlers in evaluate.py
and do NOT appear in this mapper.
"""
from . import functions as fn
import splitweave.symbolic as sws
import geolipi.symbolic as gls


sws_to_fn_mapper = {
    # ---- Base Grids ----
    sws.CartesianGrid: fn.dummy_func,
    sws.PolarToCart: fn.polar_to_cart_grid,
    sws.PolarGrid: fn.cart_to_polar_grid,
    sws.CartToPolar: fn.cart_to_polar_grid,

    # ---- Rectangular Partitions ----
    sws.RectRepeat: fn.rect_repeat_grid,
    sws.RectRepeatInner: fn.rect_repeat_grid_fitting,
    sws.RectRepeatFitting: fn.rect_repeat_grid_fitting,
    sws.RectRepeatShiftedX: fn.cart_to_brick_repeat_grid_x,
    sws.RectRepeatShiftedY: fn.cart_to_brick_repeat_grid_y,

    # ---- Rectangular Edges ----
    sws.RectRepeatEdge: fn.rect_repeat_edge_grid,
    sws.RectRepeatShiftedXEdge: fn.cart_to_brick_edge_grid_x,
    sws.RectRepeatShiftedYEdge: fn.cart_to_brick_edge_grid_y,

    # ---- Axial Partitions ----
    sws.AxialRepeatX: fn.cartesian_repeat_x_grid,
    sws.AxialRepeatY: fn.cartesian_repeat_x_grid,

    # ---- Radial / Polar Partitions ----
    sws.RadialRepeatAngular: fn.polar_repeat_angular_grid,
    sws.RadialRepeatCentered: fn.polar_repeat_centered_grid,
    sws.RadialRepeatInitRadial: fn.polar_repeat_init_radial_grid,
    sws.RadialRepeatBricked: fn.polar_repeat_bricked_grid,
    sws.RadialRepeatFixedArc: fn.polar_repeat_radial_fixed_arc_grid,
    sws.RadialRepeatFixedArcBricked: fn.polar_repeat_fixed_arc_bricked_grid,
    sws.AngularRepeat: fn.polar_repeat_angular_grid,

    # ---- Radial / Polar Edges ----
    sws.RadialRepeatEdge: fn.polar_repeat_edge_grid,
    sws.RadialRepeatBrickedEdge: fn.polar_repeat_edge_bricked_grid,
    sws.RadialRepeatFixedArcEdge: fn.polar_repeat_edge_fixed_arc,
    sws.RadialRepeatFixedArcBrickedEdge: fn.polar_repeat_edge_fixed_arc_bricked,
    sws.PolarRepeatEdge: fn.polar_repeat_edge_grid,

    # ---- Hexagonal Partitions ----
    sws.HexRepeat: fn.cart_to_hex_grid,
    sws.HexRepeatX: fn.cart_to_hex_grid,
    sws.HexRepeatY: fn.cart_to_hex_grid_flip,

    # ---- Hexagonal Edges ----
    sws.HexRepeatEdge: fn.cart_to_hex_edge_grid,
    sws.HexRepeatXEdge: fn.cart_to_hex_edge_grid,
    sws.HexRepeatYEdge: fn.cart_to_hex_edge_grid_flip,

    # ---- Triangular Partitions ----
    sws.TriangularRepeat: fn.cart_to_triangular_grid,
    sws.TriangularRepeatEdge: fn.cart_to_triangular_edge_grid,

    # ---- Diamond Partitions ----
    sws.DiamondRepeat: fn.cart_to_diamond_grid,
    sws.DiamondRepeatEdge: fn.cart_to_diamond_edge_grid,

    # ---- Voronoi / Irregular Partitions ----
    sws.VoronoiRepeat: fn.cart_to_voronoi_grid_sizewise,
    sws.VoronoiRepeatEdge: fn.cart_to_voronoi_edge_grid_sizewise,
    sws.VoronoiRepeatRadialDeformed: fn.cart_to_voronoi_radially_deformed_grid,
    sws.IrregularRectRepeat: fn.cart_to_aperiodic_box_grid,
    sws.IrregularRectRepeatEdge: fn.cart_to_aperiodic_box_edge_grid,

    # ---- Delaunay ----
    sws.DelaunayRepeat: fn.cart_to_delaunay_grid,

    # ---- Cartesian Transforms ----
    sws.CartTranslate: fn.cart_translate_grid,
    sws.CartScale: fn.cart_scale_grid,
    sws.CartRotate: fn.cart_rotate_grid,
    sws.CartAffine: fn.cart_affine_grid,

    # ---- Polar Transforms ----
    sws.PolarRotate: fn.polar_rotate_grid,
    sws.PolarScale: fn.polar_scale_grid,
    sws.PolarTranslate: fn.polar_translate_grid,

    # ---- Signal-driven Transforms ----
    sws.TranslateWtSignal: fn.translate_with_signal,
    sws.RotateWtSignal: fn.rotate_with_signal,
    sws.ScaleWtSignal: fn.scale_with_signal,
    sws.TranslateXWtSignal: fn.translate_x_with_signal,
    sws.TranslateYWtSignal: fn.translate_y_with_signal,
    sws.ScaleXWtSignal: fn.scale_x_with_signal,
    sws.ScaleYWtSignal: fn.scale_y_with_signal,

    # ---- Noise Functions ----
    sws.ValueNoise: fn.value_noise,
    sws.PerlinNoise: fn.perlin_noise,

    # ---- Utility ----
    gls.Param: fn.dummy_func,
}

import geolipi.symbolic as gls
from geolipi.symbolic.registry import register_symbol
from .layout import GridFunction


class ColorFill(GridFunction):
    """Base class for region color-filling operations (RSP patterns).
    
    ColorFill nodes take grid_ids (cell assignments) and produce
    an RGBA canvas by assigning colors to regions.
    """


@register_symbol
class SolidColorFill(ColorFill):
    """Fill each region with a solid color from a palette.

    Colors are assigned cyclically from the palette based on grid_ids.
    id_reduce: how to collapse multi-dimensional grid_ids to 1D.
    mod_k: for sum_mod, modulo divisor.
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid_ids": {"type": "Expr"},
            "palette": {"type": "List[Vector[4]]"},
            "n_colors": {"type": "int"},
            "id_reduce": {"type": 'Enum["none"|"x"|"y"|"reduce_sum"|"sum_mod"]', "optional": True},
            "mod_k": {"type": "int", "optional": True},
        }


@register_symbol
class InterpColorFill(ColorFill):
    """Fill regions by interpolating between two colors.

    Interpolation varies across cells based on their grid ID order.
    Modes: 'simple' (linear), 'symmetric' (fold at midpoint), 'siny' (sinusoidal).
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid_ids": {"type": "Expr"},
            "color_a": {"type": "Vector[4]"},
            "color_b": {"type": "Vector[4]"},
            "interp_mode": {"type": 'Enum["simple"|"symmetric"|"siny"]'},
            "sin_k": {"type": "int", "optional": True},
            "mid_point": {"type": "float", "optional": True},
            "id_reduce": {"type": 'Enum["none"|"x"|"y"|"reduce_sum"|"sum_mod"]', "optional": True},
            "mod_k": {"type": "int", "optional": True},
        }


@register_symbol
class TriInterpColorFill(ColorFill):
    """Fill regions by interpolating between three colors.

    Three-way interpolation across cells based on grid ID order.
    interp_mode: 'simple', 'symmetric', or 'siny'.
    mid_point: split point between color_a->b and color_b->c.
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid_ids": {"type": "Expr"},
            "color_a": {"type": "Vector[4]"},
            "color_b": {"type": "Vector[4]"},
            "color_c": {"type": "Vector[4]"},
            "interp_mode": {"type": 'Enum["simple"|"symmetric"|"siny"]', "optional": True},
            "sin_k": {"type": "int", "optional": True},
            "mid_point": {"type": "float", "optional": True},
            "id_reduce": {"type": 'Enum["none"|"x"|"y"|"reduce_sum"|"sum_mod"]', "optional": True},
            "mod_k": {"type": "int", "optional": True},
        }


@register_symbol
class ColorModulate(ColorFill):
    """Apply a texture modulator on top of a colored canvas.

    Modulators add visual texture within each region using geolipi 2D primitives.
    modulator: geolipi expression that evaluates to alpha mask (0-1) on layout grid.
    layout_expr: grid expression used to evaluate the modulator.
    """
    @classmethod
    def default_spec(cls):
        return {
            "canvas": {"type": "Expr"},
            "modulator": {"type": "Expr"},
            "layout_expr": {"type": "Expr"},
            "bg_color": {"type": "Vector[4]", "optional": True},
        }

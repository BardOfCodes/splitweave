import geolipi.symbolic as gls
from geolipi.symbolic.registry import register_symbol
from .layout import GridFunction


class CompositeNode(gls.GLFunction):
    """Base class for layer composition operations.
    
    Composite nodes combine multiple RGBA canvases into a final output.
    Note: SourceOverSequence from geolipi is reused for alpha compositing.
    """


# ---- Backgrounds (eval to full canvas [N, 4]) ----

@register_symbol
class ConstantBackground(CompositeNode):
    """Solid-color background. Evaluates to a full-resolution RGBA canvas (ones * color).
    
    Args: color (RGBA tensor [4])
    """
    @classmethod
    def default_spec(cls):
        return {
            "color": {"type": "Vector[4]"},
        }


@register_symbol
class ApplyColoring(CompositeNode):
    """
    Background coloring applied over a layout grid.
    Evaluates to a canvas [N, 4] with per-cell hue-shifted colors.

    This is the grid_n_color background from patternator:
    1. Evaluate layout_expr to get grid_ids
    2. Compute a discrete signal from grid_ids using the signal expression
    3. Shift the base_color hue per cell using recolor_params

    Args:
        layout_expr: GridFunction expression (layout for the background grid)
        base_color: RGBA tensor [4] (base background color)
        recolor_params: tuple (recolor_type, recolor_seed, signal_mode)
            - recolor_type: 0=smooth, 1=discrete
            - recolor_seed: int seed for random hue generation
            - signal_mode: 0=single, 1=double
        signal: typed discrete signal expression (CheckerboardSignal, XStripeSignal, etc.)
    """
    @classmethod
    def default_spec(cls):
        return {
            "layout": {"type": "Expr"},
            "base_color": {"type": "Vector[4]"},
            "recolor_params": {"type": "Vector[3]"},
            "signal": {"type": "Expr"},
        }


# ---- Legacy symbols (kept for backward compat) ----

@register_symbol
class BackgroundSolid(CompositeNode):
    """A solid-color background canvas (legacy alias for ConstantBackground)."""
    @classmethod
    def default_spec(cls):
        return {
            "color": {"type": "Vector[4]"},
        }


@register_symbol
class BackgroundGrid(CompositeNode):
    """A grid-based colored background (legacy).
    
    Uses grid_ids and a signal to create spatially varying
    background colors, typically via hue shifts.
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid_ids": {"type": "Expr"},
            "signal": {"type": "Expr"},
            "recolor_type": {"type": 'Enum["smooth"|"discrete"]'},
            "recolor_seed": {"type": "int"},
            "color_seed": {"type": "int"},
        }


# ---- Borders ----

@register_symbol
class BorderEffect(CompositeNode):
    """Render borders along cell edges.
    
    Takes the border grid expression (from edge partition symbols)
    and renders it with the specified mode and color.
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid_ids": {"type": "Expr"},
            "border_expr": {"type": "Expr"},
            "border_mode": {"type": 'Enum["simple"|"dotted"|"onion"|"siny"]'},
            "border_size": {"type": "float"},
            "color": {"type": "Vector[4]"},
            "signal": {"type": "Expr", "optional": True},
        }


# ---- Fill Mask ----

@register_symbol
class FillMask(CompositeNode):
    """Fill a masked region with tiles from a grid layout.
    
    Used for secondary fill passes where tiles are placed only
    in regions defined by a mask.
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid_expr": {"type": "Expr"},
            "tile_expr": {"type": "Expr"},
            "mask": {"type": "Expr"},
            "fill_mode": {"type": 'Enum["outside"|"inside_border"|"inside"|"outside_border"]'},
            "stochastic_drop": {"type": "bool"},
            "drop_rate": {"type": "float"},
            "normalize": {"type": "bool"},
            "norm_mode": {"type": "int"},
        }

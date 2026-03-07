import geolipi.symbolic as gls
from geolipi.symbolic.registry import register_symbol
from .layout import GridFunction


class LayoutCellEffect(GridFunction):
    """Base for cell effects that wrap a layout and modify the grid at eval time.
    Takes (layout_expr, signal_expr, ...effect_params). signal_expr is a typed
    discrete signal (CheckerboardSignal, XStripeSignal, etc.)."""
    pass


@register_symbol
class LayoutCellTranslate(LayoutCellEffect):
    """Translate grid per cell; wraps layout_expr. signal: typed discrete signal expr."""
    @classmethod
    def default_spec(cls):
        return {
            "layout": {"type": "Expr"},
            "signal": {"type": "Expr"},
            "t_x": {"type": "float"},
            "t_y": {"type": "float"},
            "mode": {"type": 'Enum["single"|"double"]'},
        }


@register_symbol
class LayoutCellRotate(LayoutCellEffect):
    """Rotate grid per cell; wraps layout_expr."""
    @classmethod
    def default_spec(cls):
        return {
            "layout": {"type": "Expr"},
            "signal": {"type": "Expr"},
            "rotation": {"type": "float"},
            "mode": {"type": 'Enum["single"|"double"]'},
        }


@register_symbol
class LayoutCellScale(LayoutCellEffect):
    """Scale grid per cell; wraps layout_expr."""
    @classmethod
    def default_spec(cls):
        return {
            "layout": {"type": "Expr"},
            "signal": {"type": "Expr"},
            "scale": {"type": "float"},
            "mode": {"type": 'Enum["single"]'},
        }


@register_symbol
class LayoutCellReflect(LayoutCellEffect):
    """Reflect grid per cell along x, y, or xy; wraps layout_expr."""
    @classmethod
    def default_spec(cls):
        return {
            "layout": {"type": "Expr"},
            "signal": {"type": "Expr"},
            "reflect": {"type": 'Enum["x"|"y"|"xy"]'},
            "mode": {"type": 'Enum["single"]'},
        }


class CellEffect(GridFunction):
    """Base class for per-cell effects.
    
    Cell effects operate on a per-cell basis using grid_ids to identify
    which cell each pixel belongs to, and a signal to control the
    effect intensity/selection per cell.
    """


class CellGridEffect(CellEffect):
    """Base class for cell effects that modify UV grid coordinates.
    
    These are applied BEFORE tile evaluation (spatial transforms).
    Takes (grid, grid_ids, signal, ...params) -> modified grid.
    """


class CellCanvasEffect(CellEffect):
    """Base class for cell effects that modify the RGBA canvas.
    
    These are applied AFTER tile evaluation (visual effects).
    Takes (canvas, grid_ids, signal, ...params) -> modified canvas.
    """


# ---- Grid Effects (spatial, pre-evaluation) ----

@register_symbol
class CellTranslate(CellGridEffect):
    """Translate grid coordinates per cell, driven by a signal.
    
    In 'single' mode, applies t_x/t_y to cells where signal is active.
    In 'double' mode, applies t_x/t_y and -t_x/-t_y alternately.
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "grid_ids": {"type": "Expr"},
            "signal": {"type": "Expr"},
            "t_x": {"type": "float"},
            "t_y": {"type": "float"},
            "mode": {"type": 'Enum["single"|"double"]'},
        }


@register_symbol
class CellRotate(CellGridEffect):
    """Rotate grid coordinates per cell, driven by a signal."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "grid_ids": {"type": "Expr"},
            "signal": {"type": "Expr"},
            "rotation": {"type": "float"},
            "mode": {"type": 'Enum["single"|"double"]'},
        }


@register_symbol
class CellScale(CellGridEffect):
    """Scale grid coordinates per cell, driven by a signal."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "grid_ids": {"type": "Expr"},
            "signal": {"type": "Expr"},
            "scale": {"type": "float"},
            "mode": {"type": 'Enum["single"]'},
        }


@register_symbol
class CellReflect(CellGridEffect):
    """Reflect grid coordinates per cell along x, y, or both axes."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "grid_ids": {"type": "Expr"},
            "signal": {"type": "Expr"},
            "reflect": {"type": 'Enum["x"|"y"|"xy"]'},
            "mode": {"type": 'Enum["single"]'},
        }


# ---- Tile Ordering (bridges grid -> canvas) ----

@register_symbol
class TileOrder(CellCanvasEffect):
    """Select which tile to render per cell based on a signal.
    
    Given N tile expressions, uses the signal to assign a tile
    index to each cell and evaluates the corresponding tile.
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "grid_ids": {"type": "Expr"},
            "signal": {"type": "Expr"},
            "n_tiles": {"type": "int"},
            "tile_1": {"type": "Expr"},
            "tile_2": {"type": "Expr", "optional": True},
            "tile_3": {"type": "Expr", "optional": True},
            "tile_4": {"type": "Expr", "optional": True},
        }


# ---- Canvas Effects (visual, post-evaluation) ----

@register_symbol
class CellRecolor(CellCanvasEffect):
    """Shift colors per cell, driven by a signal.
    
    'smooth' mode: continuous hue shift proportional to signal value.
    'discrete' mode: discrete color assignments per cell.
    """
    @classmethod
    def default_spec(cls):
        return {
            "canvas": {"type": "Expr"},
            "grid_ids": {"type": "Expr"},
            "signal": {"type": "Expr"},
            "recolor_type": {"type": 'Enum["smooth"|"discrete"]'},
            "recolor_seed": {"type": "int"},
            "mode": {"type": 'Enum["single"|"double"]'},
        }


@register_symbol
class CellOutline(CellCanvasEffect):
    """Add outlines to cells where the signal is active."""
    @classmethod
    def default_spec(cls):
        return {
            "canvas": {"type": "Expr"},
            "grid_ids": {"type": "Expr"},
            "signal": {"type": "Expr"},
            "outline_color": {"type": "Vector[4]"},
            "outline_thickness": {"type": "float"},
            "mode": {"type": 'Enum["single"]'},
        }


@register_symbol
class CellShadow(CellCanvasEffect):
    """Add drop shadows to cells where the signal is active."""
    @classmethod
    def default_spec(cls):
        return {
            "canvas": {"type": "Expr"},
            "grid": {"type": "Expr"},
            "grid_ids": {"type": "Expr"},
            "signal": {"type": "Expr"},
            "shadow_thickness": {"type": "float"},
            "shift_x": {"type": "float"},
            "shift_y": {"type": "float"},
            "mode": {"type": 'Enum["single"]'},
        }


@register_symbol
class CellOpacity(CellCanvasEffect):
    """Modulate opacity per cell, driven by a signal."""
    @classmethod
    def default_spec(cls):
        return {
            "canvas": {"type": "Expr"},
            "grid_ids": {"type": "Expr"},
            "signal": {"type": "Expr"},
            "opacity": {"type": "float"},
            "mode": {"type": 'Enum["single"]'},
        }


# ---- Pattern-level canvas effect wrappers (apply after tile eval) ----
# These wrap ApplyTile/ApplyMultiTile and apply Recolor/Outline/Opacity to the canvas.
# signal is a typed discrete signal expression (CheckerboardSignal, XStripeSignal, etc.).


@register_symbol
class ApplyCellRecolor(gls.GLFunction):
    """Wrap a pattern expr; after evaluation apply per-cell recolor (HSL hue shift)."""
    @classmethod
    def default_spec(cls):
        return {
            "pattern_expr": {"type": "Expr"},
            "signal": {"type": "Expr"},
            "recolor_type": {"type": 'Enum["smooth"|"discrete"]'},
            "recolor_seed": {"type": "int"},
            "mode": {"type": 'Enum["single"|"double"]'},
        }


@register_symbol
class ApplyCellOutline(gls.GLFunction):
    """Wrap a pattern expr; after evaluation apply per-cell outline."""
    @classmethod
    def default_spec(cls):
        return {
            "pattern_expr": {"type": "Expr"},
            "signal": {"type": "Expr"},
            "outline_color": {"type": "Vector[4]"},
            "outline_thickness": {"type": "float"},
            "mode": {"type": 'Enum["single"]'},
        }


@register_symbol
class ApplyCellOpacity(gls.GLFunction):
    """Wrap a pattern expr; after evaluation apply per-cell opacity modulation."""
    @classmethod
    def default_spec(cls):
        return {
            "pattern_expr": {"type": "Expr"},
            "signal": {"type": "Expr"},
            "opacity": {"type": "float"},
            "mode": {"type": 'Enum["single"]'},
        }

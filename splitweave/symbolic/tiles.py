import geolipi.symbolic as gls
from geolipi.symbolic.registry import register_symbol

from .layout import GridFunction


class TileNode(gls.GLFunction):
    """Base class for tile-related operations.
    
    Tiles are raster images (RGBA) that are placed into grid cells.
    TileNode subclasses either define a tile source or apply
    per-tile visual effects.
    """


class TileEffect(TileNode):
    """Base class for per-tile visual effects.
    
    Each effect takes a tile expression as the first argument
    and returns a modified tile expression.
    """


@register_symbol
class TileSource(TileNode):
    """A tile image source.
    
    Wraps a TileUV2D or image tensor as a tile that can be
    composed with tile effects.
    """
    @classmethod
    def default_spec(cls):
        return {
            "source": {"type": "Expr"},
        }


@register_symbol
class ApplyTile(gls.GLFunction):
    """Root pattern expression: apply a tile to a layout.
    
    Single expression that can be executed to produce (canvas, grid_ids).
    layout: GridFunction (e.g. RectRepeat(CartesianGrid(), size))
    tile: GLFunction (e.g. TileUV2D(tensor)) evaluated at layout coords.
    """
    @classmethod
    def default_spec(cls):
        return {
            "layout": {"type": "Expr"},
            "tile": {"type": "Expr"},
        }


@register_symbol
class ApplyMultiTile(gls.GLFunction):
    """Root pattern expression: apply multiple tiles to a layout with per-cell tile selection.
    
    signal: typed discrete signal expression (CheckerboardSignal, XStripeSignal, etc.).
    Per-cell discrete signal is computed from grid_ids; value in 0..n_tiles-1
    selects which tile to render at that cell.
    """
    @classmethod
    def default_spec(cls):
        return {
            "layout": {"type": "Expr"},
            "signal": {"type": "Expr"},
            "tile_1": {"type": "Expr"},
            "tile_2": {"type": "Expr", "optional": True},
            "tile_3": {"type": "Expr", "optional": True},
            "tile_4": {"type": "Expr", "optional": True},
        }


@register_symbol
class TileRecolor(TileEffect):
    """Shift the hue of a tile.
    
    Converts tile to HSL, shifts hue by the given amount, converts back.
    """
    @classmethod
    def default_spec(cls):
        return {
            "tile": {"type": "Expr"},
            "hue": {"type": "float"},
        }


@register_symbol
class TileOutline(TileEffect):
    """Add an outline around the tile's visible region."""
    @classmethod
    def default_spec(cls):
        return {
            "tile": {"type": "Expr"},
            "thickness": {"type": "float"},
            "color": {"type": "Vector[4]"},
        }


@register_symbol
class TileShadow(TileEffect):
    """Add a drop shadow behind the tile."""
    @classmethod
    def default_spec(cls):
        return {
            "tile": {"type": "Expr"},
            "thickness": {"type": "float"},
        }


@register_symbol
class TileRotate(TileEffect):
    """Rotate the tile by a given angle (radians)."""
    @classmethod
    def default_spec(cls):
        return {
            "tile": {"type": "Expr"},
            "angle": {"type": "float"},
        }


@register_symbol
class TileScale(TileEffect):
    """Scale the tile uniformly."""
    @classmethod
    def default_spec(cls):
        return {
            "tile": {"type": "Expr"},
            "scale": {"type": "float"},
        }


@register_symbol
class TileReflectX(TileEffect):
    """Reflect the tile along the X axis."""
    @classmethod
    def default_spec(cls):
        return {
            "tile": {"type": "Expr"},
        }


@register_symbol
class TileReflectY(TileEffect):
    """Reflect the tile along the Y axis."""
    @classmethod
    def default_spec(cls):
        return {
            "tile": {"type": "Expr"},
        }


@register_symbol
class TileOpacity(TileEffect):
    """Modify the opacity (alpha) of the tile."""
    @classmethod
    def default_spec(cls):
        return {
            "tile": {"type": "Expr"},
            "opacity": {"type": "float"},
        }

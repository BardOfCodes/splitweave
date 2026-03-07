import geolipi.symbolic as gls
from geolipi.symbolic.registry import register_symbol
from .layout import GridFunction


# ---- Grid Transforms (constant parameter) ----

@register_symbol
class TransformGrid(GridFunction):
    """Base class for constant-parameter grid transforms."""


@register_symbol
class CartTranslate(TransformGrid):
    """Translate grid in Cartesian coordinates."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "offset": {"type": "Vector[2]"},
        }


@register_symbol
class CartScale(TransformGrid):
    """Scale grid in Cartesian coordinates."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "scale": {"type": "Vector[2]"},
        }


@register_symbol
class CartRotate(TransformGrid):
    """Rotate grid in Cartesian coordinates."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "angle": {"type": "float"},
        }


@register_symbol
class CartAffine(TransformGrid):
    """Apply a full affine transformation matrix to the grid."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "matrix": {"type": "Matrix[3,3]"},
        }


@register_symbol
class PolarRotate(TransformGrid):
    """Rotate grid in polar coordinates (add to theta)."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "angle": {"type": "float"},
        }


@register_symbol
class PolarScale(TransformGrid):
    """Scale grid in polar coordinates (multiply r)."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "scale": {"type": "float"},
        }


@register_symbol
class PolarTranslate(TransformGrid):
    """Translate grid in polar coordinates."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "offset": {"type": "Vector[2]"},
        }


# ---- Signal-driven Transforms ----

@register_symbol
class SignalTransformGrid(GridFunction):
    """Base class for signal-driven grid transforms.
    
    These apply spatially-varying transformations where the
    amount is controlled by a signal field.
    """


@register_symbol
class TranslateWtSignal(SignalTransformGrid):
    """Translate grid by a 2D signal field (additive offset)."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "signal": {"type": "Expr"},
        }


@register_symbol
class RotateWtSignal(SignalTransformGrid):
    """Rotate grid by a scalar signal field (per-point angle)."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "signal": {"type": "Expr"},
        }


@register_symbol
class ScaleWtSignal(SignalTransformGrid):
    """Scale grid by a scalar signal field."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "signal": {"type": "Expr"},
        }


@register_symbol
class TranslateXWtSignal(SignalTransformGrid):
    """Translate grid X by a scalar signal field."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "signal": {"type": "Expr"},
        }


@register_symbol
class TranslateYWtSignal(SignalTransformGrid):
    """Translate grid Y by a scalar signal field."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "signal": {"type": "Expr"},
        }


@register_symbol
class ScaleXWtSignal(SignalTransformGrid):
    """Scale grid X by a scalar signal field."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "signal": {"type": "Expr"},
        }


@register_symbol
class ScaleYWtSignal(SignalTransformGrid):
    """Scale grid Y by a scalar signal field."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "signal": {"type": "Expr"},
        }


# ---- Scalar Fields / Noise ----

@register_symbol
class Scalar2D(GridFunction):
    """Base class for 2D scalar field generators."""


@register_symbol
class ValueNoise(Scalar2D):
    """Value noise scalar field."""
    @classmethod
    def default_spec(cls):
        return {
            "resolution": {"type": "float"},
        }


@register_symbol
class PerlinNoise(Scalar2D):
    """Perlin gradient noise scalar field."""
    @classmethod
    def default_spec(cls):
        return {
            "resolution": {"type": "float"},
        }


# ---- Utility ----

@register_symbol
class GridBundle(gls.GLFunction):
    """Bundle a precomputed grid and optional grid_ids into a single node.
    
    Used to inject pre-evaluated tensors into the expression tree.
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Tensor"},
            "grid_ids": {"type": "Tensor", "optional": True},
        }

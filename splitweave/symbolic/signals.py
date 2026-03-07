import geolipi.symbolic as gls
from geolipi.symbolic.registry import register_symbol
from .layout import GridFunction


class Signal2D(GridFunction):
    """Base class for 2D scalar signal fields.
    
    Signals produce a scalar field over the canvas that can be used
    to drive cell effects, deformations, and color modulation.
    Unlike Scalar2D (raw noise), Signal2D represents parametric
    signal functions with specific spatial structure.
    """


class ContinuousSignal(Signal2D):
    """Base class for continuous (per-pixel) signals."""


class DiscreteSignalBase(Signal2D):
    """Base class for discrete (per-cell) signals."""


@register_symbol
class RadialSignal(ContinuousSignal):
    """Radial distance signal from a center point.
    
    Computes distance from center, then applies a transfer function
    (linear, siny, sigmoid, exponential) to produce a 0-1 scalar field.
    """
    @classmethod
    def default_spec(cls):
        return {
            "center": {"type": "Vector[2]"},
            "signal_mode": {"type": 'Enum["linear"|"siny"|"sigmoid"|"exponential"]'},
            "sin_k": {"type": "float", "optional": True},
            "phase_shift": {"type": "float", "optional": True},
            "sigmoid_rate": {"type": "float", "optional": True},
            "sigmoid_spread": {"type": "float", "optional": True},
        }


@register_symbol
class PerlinSignal(ContinuousSignal):
    """Multi-octave Perlin noise signal.
    
    Sums multiple Perlin noise octaves at different resolutions
    to produce a smooth random scalar field.
    """
    @classmethod
    def default_spec(cls):
        return {
            "resolution_seq": {"type": "Tuple[int]"},
            "seed": {"type": "int"},
        }


@register_symbol
class DecaySignal(ContinuousSignal):
    """Directional linear decay signal.
    
    Projects coordinates onto a direction vector to produce a
    monotonically varying scalar field.
    """
    @classmethod
    def default_spec(cls):
        return {
            "direction": {"type": "Vector[2]"},
            "axis": {"type": 'Enum["x"|"y"|"xy"]'},
        }


@register_symbol
class StripSignal(ContinuousSignal):
    """Sinusoidal strip signal along an angle.
    
    Produces a periodic scalar field: sin(proj * sin_k * pi + phase_shift).
    """
    @classmethod
    def default_spec(cls):
        return {
            "angle": {"type": "float"},
            "sin_k": {"type": "float"},
            "phase_shift": {"type": "float"},
        }


@register_symbol
class SwirlSignal(ContinuousSignal):
    """Radial signal variant used for swirl deformations.
    
    Same as RadialSignal but with additional sigmoid mode support
    for swirl-specific parameterization.
    """
    @classmethod
    def default_spec(cls):
        return {
            "center": {"type": "Vector[2]"},
            "signal_mode": {"type": 'Enum["linear"|"siny"|"sigmoid"]'},
            "sin_k": {"type": "float", "optional": True},
            "phase_shift": {"type": "float", "optional": True},
            "sigmoid_rate": {"type": "float", "optional": True},
            "sigmoid_spread": {"type": "float", "optional": True},
        }


# ---- Typed discrete signals ----
# Shared params (minimal for evaluation): k, inverse, group_alternate, apply_sym, double_dip.
# Mode is determined by the signal class; DiagonalSignal and CountSignal add axis.


def _discrete_signal_common_spec():
    """Shared spec for all typed discrete signals. Only params used in evaluation."""
    return {
        "k": {"type": "int"},
        "inverse": {"type": "bool", "optional": True},
        "group_alternate": {"type": "bool", "optional": True},
        "apply_sym": {"type": "bool", "optional": True},
        "double_dip": {"type": "bool", "optional": True},
        "normalize": {"type": "bool", "optional": True},
    }


@register_symbol
class CheckerboardSignal(DiscreteSignalBase):
    """Checkerboard pattern: (ix + iy) % k."""
    @classmethod
    def default_spec(cls):
        return dict(**_discrete_signal_common_spec())


@register_symbol
class XStripeSignal(DiscreteSignalBase):
    """Horizontal stripes: ix % k."""
    @classmethod
    def default_spec(cls):
        return dict(**_discrete_signal_common_spec())


@register_symbol
class YStripeSignal(DiscreteSignalBase):
    """Vertical stripes: iy % k."""
    @classmethod
    def default_spec(cls):
        return dict(**_discrete_signal_common_spec())


@register_symbol
class XXStripeSignal(DiscreteSignalBase):
    """X-based stripes with clip: clip(ix % (k+1), 0, k)."""
    @classmethod
    def default_spec(cls):
        return dict(**_discrete_signal_common_spec())


@register_symbol
class YYStripeSignal(DiscreteSignalBase):
    """Y-based stripes with clip: clip(iy % (k+1), 0, k)."""
    @classmethod
    def default_spec(cls):
        return dict(**_discrete_signal_common_spec())


@register_symbol
class XYStripeSignal(DiscreteSignalBase):
    """Combined XY: min(ix % k, iy % k)."""
    @classmethod
    def default_spec(cls):
        return dict(**_discrete_signal_common_spec())


@register_symbol
class DiagonalSignal(DiscreteSignalBase):
    """Diagonal pattern. axis: 'diag_1' (ix+iy) or 'diag_2' (iy-ix)."""
    @classmethod
    def default_spec(cls):
        spec = dict(**_discrete_signal_common_spec())
        spec["axis"] = {"type": 'Enum["diag_1"|"diag_2"]'}
        return spec


@register_symbol
class CountSignal(DiscreteSignalBase):
    """Count-based signal. axis: 'count_diag_1'|'count_x'|'count_y'|'count_diag_2'."""
    @classmethod
    def default_spec(cls):
        spec = dict(**_discrete_signal_common_spec())
        spec["axis"] = {"type": 'Enum["count_diag_1"|"count_x"|"count_y"|"count_diag_2"]'}
        return spec


@register_symbol
class RandomSignal(DiscreteSignalBase):
    """Random assignment per unique cell, k levels."""
    @classmethod
    def default_spec(cls):
        return dict(**_discrete_signal_common_spec())


@register_symbol
class FullyRandomSignal(DiscreteSignalBase):
    """Fully random per unique cell (continuous)."""
    @classmethod
    def default_spec(cls):
        return dict(**_discrete_signal_common_spec())

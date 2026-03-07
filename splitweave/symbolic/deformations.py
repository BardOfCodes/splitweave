import geolipi.symbolic as gls
from geolipi.symbolic.registry import register_symbol
from .layout import GridFunction


class DeformGrid(GridFunction):
    """Base class for grid deformation operations.
    
    Deformations warp the UV grid space based on a parametric signal.
    All deformations take a grid expression as the first argument.
    """


@register_symbol
class RadialDeform(DeformGrid):
    """Scale grid by a radial distance signal from a center point.
    
    grid_out = grid * (1 + signal(||p - center||) * dist_rate)
    where signal is one of: linear, siny, sigmoid.
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "center": {"type": "Vector[2]"},
            "signal_mode": {"type": 'Enum["linear"|"siny"|"sigmoid"]'},
            "dist_rate": {"type": "float"},
            "sin_k": {"type": "float", "optional": True},
            "phase_shift": {"type": "float", "optional": True},
            "sigmoid_rate": {"type": "float", "optional": True},
            "sigmoid_spread": {"type": "float", "optional": True},
        }


@register_symbol
class PerlinDeform(DeformGrid):
    """Offset grid by multi-octave Perlin noise.
    
    grid_out = grid + noise_vector * dist_rate
    where noise_vector is sampled from summed Perlin octaves.
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "resolution_seq": {"type": "Tuple[int]"},
            "seed": {"type": "int"},
            "dist_mode": {"type": 'Enum["x"|"y"|"xy"]'},
            "dist_rate": {"type": "float"},
        }


@register_symbol
class DecayDeform(DeformGrid):
    """Scale grid by a directional linear decay.
    
    grid_out = grid * decay_factor(direction, axis)
    where decay_factor varies linearly along the given direction.
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "direction": {"type": "Vector[2]"},
            "axis": {"type": 'Enum["x"|"y"|"xy"]'},
            "dist_rate": {"type": "float"},
        }


@register_symbol
class StripDeform(DeformGrid):
    """Offset grid by sinusoidal strips at a given angle.
    
    grid_out = grid + sin(projected_coord * sin_k * pi + phase_shift) * dist_rate
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "angle": {"type": "float"},
            "axis": {"type": 'Enum["x"|"y"|"xy"]'},
            "sin_k": {"type": "float"},
            "phase_shift": {"type": "float"},
            "dist_rate": {"type": "float"},
        }


@register_symbol
class SwirlDeform(DeformGrid):
    """Rotate grid points around a center by a radially-varying angle.
    
    Translates to center, rotates by signal(||p - center||) * dist_rate,
    translates back.
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "center": {"type": "Vector[2]"},
            "signal_mode": {"type": 'Enum["linear"|"siny"|"sigmoid"]'},
            "dist_rate": {"type": "float"},
            "sin_k": {"type": "float", "optional": True},
            "phase_shift": {"type": "float", "optional": True},
            "sigmoid_rate": {"type": "float", "optional": True},
            "sigmoid_spread": {"type": "float", "optional": True},
        }


@register_symbol
class NoDeform(DeformGrid):
    """Identity deformation -- passes grid through unchanged."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
        }

"""
Color utilities for pattern generation.
Tetradic color scheme: n hues evenly spaced around the color wheel.
"""
import colorsys
import numpy as np


class _HSLColor:
    """Simple HSL color with lightness and to_rgb() for compatibility with rsp_color."""

    def __init__(self, h: float, s: float, l: float):
        self.hue = h
        self.saturation = s
        self.lightness = l

    def to_rgb(self) -> np.ndarray:
        """Return RGB as numpy array in [0, 1] range."""
        r, g, b = colorsys.hls_to_rgb(self.hue, self.lightness, self.saturation)
        return np.array([r, g, b], dtype=np.float64)


def tetradic_generator(n: int):
    """
    Generate n colors with hues evenly spaced around the color wheel (tetradic scheme).
    For n=4, uses 0°, 90°, 180°, 270°. For other n, spaces hues at 360/n degrees.

    Returns a list of color objects with .lightness and .to_rgb() methods.
    """
    base_hue = np.random.uniform(0, 1)
    saturation = np.random.uniform(0.5, 1.0)
    lightness = np.random.uniform(0.4, 0.7)

    colors = []
    for i in range(n):
        h = (base_hue + i / n) % 1.0
        c = _HSLColor(h, saturation, lightness)
        colors.append(c)
    return colors

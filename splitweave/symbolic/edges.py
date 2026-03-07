import geolipi.symbolic as gls
from geolipi.symbolic.registry import register_symbol
from .layout import PartitionGrid

######### EDGES
# Edge symbols mirror their partition counterparts but produce
# edge/border distance fields instead of cell coordinates.
# They share the same default_spec as their partition counterpart.

@register_symbol
class RectRepeatEdge(PartitionGrid):
    """Edge field for rectangular grid."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "size": {"type": "Vector[2]"},
        }


@register_symbol
class RectRepeatShiftedXEdge(PartitionGrid):
    """Edge field for X-shifted brick pattern."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "size": {"type": "Vector[2]"},
            "x_shift": {"type": "float"},
        }


@register_symbol
class RectRepeatShiftedYEdge(PartitionGrid):
    """Edge field for Y-shifted brick pattern."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "size": {"type": "Vector[2]"},
            "y_shift": {"type": "float"},
        }


@register_symbol
class DiamondRepeatEdge(PartitionGrid):
    """Edge field for diamond grid."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "grid_size": {"type": "float"},
        }


@register_symbol
class TriangularRepeatEdge(PartitionGrid):
    """Edge field for triangular grid."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "grid_size": {"type": "float"},
        }


@register_symbol
class HexRepeatEdge(PartitionGrid):
    """Edge field for hexagonal grid."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "grid_size": {"type": "float"},
        }


@register_symbol
class HexRepeatXEdge(PartitionGrid):
    """Edge field for X-oriented hexagonal grid."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "grid_size": {"type": "float"},
        }


@register_symbol
class HexRepeatYEdge(PartitionGrid):
    """Edge field for Y-oriented hexagonal grid."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "grid_size": {"type": "float"},
        }


@register_symbol
class RadialRepeatEdge(PartitionGrid):
    """Edge field for radial grid.
    
    Maps to polar_repeat_edge_grid(grid, radial_unit, angular_unit, init_gap).
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "cell_size": {"type": "Vector[2]"},
            "init_gap": {"type": "float"},
        }


@register_symbol
class RadialRepeatFixedArcEdge(PartitionGrid):
    """Edge field for radial grid with fixed arc length.
    
    Maps to polar_repeat_edge_fixed_arc(grid, radial_unit, arc_length, init_gap).
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "cell_size": {"type": "Vector[2]"},
            "init_gap": {"type": "float"},
        }

    
@register_symbol
class RadialRepeatBrickedEdge(PartitionGrid):
    """Edge field for bricked radial grid.
    
    Maps to polar_repeat_edge_bricked_grid(grid, radial_unit, angular_unit, init_gap).
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "cell_size": {"type": "Vector[2]"},
            "init_gap": {"type": "float"},
        }


@register_symbol
class RadialRepeatFixedArcBrickedEdge(PartitionGrid):
    """Edge field for bricked radial grid with fixed arc length.
    
    Maps to polar_repeat_edge_fixed_arc_bricked(grid, radial_unit, arc_length, init_gap).
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "cell_size": {"type": "Vector[2]"},
            "init_gap": {"type": "float"},
        }


@register_symbol
class PolarRepeatEdge(PartitionGrid):
    """Edge field for polar grid.
    
    Maps to polar_repeat_edge_grid(grid, radial_unit, angular_unit, init_gap).
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "cell_size": {"type": "Vector[2]"},
            "init_gap": {"type": "float"},
        }


@register_symbol
class VoronoiRepeatEdge(PartitionGrid):
    """Edge field for Voronoi partitioning."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "cell_size": {"type": "Vector[2]"},
            "noise_rate": {"type": "float"},
        }


@register_symbol
class IrregularRectRepeatEdge(PartitionGrid):
    """Edge field for aperiodic rectangular partitioning.
    Note: the evaluator automatically injects the base coordinate grid
    (simple_grid) required by the underlying function."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "box_size": {"type": "Vector[2]"},
            "noise_rate": {"type": "float"},
        }


@register_symbol
class DelaunayRepeatEdge(PartitionGrid):
    """Edge field for Delaunay triangulation partitioning."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "cell_size": {"type": "Vector[2]"},
        }

import geolipi.symbolic as gls
from geolipi.symbolic.registry import register_symbol


class GridFunction(gls.GLFunction):
    """Base class for all grid-space operations in splitweave."""


## Base instantiation

class InstantiateGrid(GridFunction):
    """Base class for grid instantiation (creates a coordinate space)."""


@register_symbol
class CartesianGrid(InstantiateGrid):
    """Create a 2D Cartesian coordinate grid."""
    @classmethod
    def default_spec(cls):
        return {}


@register_symbol
class PolarGrid(InstantiateGrid):
    """Create a 2D polar coordinate grid (r, theta)."""
    @classmethod
    def default_spec(cls):
        return {}


#### BASE Conversions

class ConvertGrid(GridFunction):
    """Base class for coordinate system conversions."""


@register_symbol
class CartToPolar(ConvertGrid):
    """Convert Cartesian grid coordinates to polar (r, theta)."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
        }

    
@register_symbol
class PolarToCart(ConvertGrid):
    """Convert polar grid coordinates (r, theta) to Cartesian."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
        }


#### PARTITIONING
class PartitionGrid(GridFunction):
    """Base class for grid partitioning operations.
    
    Partitions divide the grid into cells, returning both the
    local coordinates within each cell and cell IDs.
    """


# ---- Rectangular partitions ----

@register_symbol
class RectRepeat(PartitionGrid):
    """Regular rectangular grid repeat."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "size": {"type": "Vector[2]"},
        }


@register_symbol
class RectRepeatInner(PartitionGrid):
    """Rectangular grid repeat (inner cells only, excludes border)."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "size": {"type": "Vector[2]"},
        }


@register_symbol
class RectRepeatFitting(PartitionGrid):
    """Rectangular grid repeat with aspect-ratio fitting."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "size": {"type": "Vector[2]"},
        }

    
@register_symbol
class RectRepeatShiftedX(PartitionGrid):
    """Brick pattern: rectangular repeat with alternating X shift."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "size": {"type": "Vector[2]"},
            "x_shift": {"type": "float"},
        }


@register_symbol
class RectRepeatShiftedY(PartitionGrid):
    """Brick pattern: rectangular repeat with alternating Y shift."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "size": {"type": "Vector[2]"},
            "y_shift": {"type": "float"},
        }


# ---- Triangular partitions ----

@register_symbol
class TriangularRepeat(PartitionGrid):
    """Triangular grid tiling."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "grid_size": {"type": "float"},
        }


@register_symbol
class TriangularBrickRepeatX(PartitionGrid):
    """Triangular grid with brick-like X offset."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "grid_size": {"type": "float"},
        }


# ---- Diamond partitions ----

@register_symbol
class DiamondRepeat(PartitionGrid):
    """Diamond (45-degree rotated square) grid tiling."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "grid_size": {"type": "float"},
        }


@register_symbol
class DiamondBrickRepeatX(PartitionGrid):
    """Diamond grid with brick-like X offset."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "grid_size": {"type": "float"},
        }


@register_symbol
class DiamondBrickRepeatY(PartitionGrid):
    """Diamond grid with brick-like Y offset."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "grid_size": {"type": "float"},
        }


# ---- Hexagonal partitions ----

@register_symbol
class HexRepeat(PartitionGrid):
    """Hexagonal grid tiling."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "grid_size": {"type": "float"},
        }


@register_symbol
class HexRepeatX(PartitionGrid):
    """Hexagonal grid tiling (X-oriented, same as HexRepeat)."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "grid_size": {"type": "float"},
        }


@register_symbol
class HexRepeatY(PartitionGrid):
    """Hexagonal grid tiling (Y-oriented / flipped)."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "grid_size": {"type": "float"},
        }


# ---- Axial partitions ----

@register_symbol
class AxialRepeatX(PartitionGrid):
    """1D repeat along X axis only."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "width": {"type": "float"},
        }


@register_symbol
class AxialRepeatY(PartitionGrid):
    """1D repeat along Y axis only."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "height": {"type": "float"},
        }

    
# ---- Radial / Polar partitions ----

@register_symbol
class RadialRepeatAngular(PartitionGrid):
    """Radial partition by angular sectors.
    
    Maps to polar_repeat_angular_grid(grid, angular_unit).
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "angular_unit": {"type": "float"},
        }


@register_symbol
class RadialRepeatInitRadial(PartitionGrid):
    """Radial partition initialized in radial coordinates.
    
    Maps to polar_repeat_init_radial_grid(grid, radial_unit, angular_unit, init_gap).
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "cell_size": {"type": "Vector[2]"},
            "init_gap": {"type": "float"},
        }


@register_symbol
class RadialRepeatCentered(PartitionGrid):
    """Radial partition centered at a specific point.
    
    Maps to polar_repeat_centered_grid(grid, radial_unit, angular_unit).
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "cell_size": {"type": "Vector[2]"},
        }


@register_symbol
class RadialRepeatFixedArc(PartitionGrid):
    """Radial partition with fixed arc length.
    
    Maps to polar_repeat_radial_fixed_arc_grid(grid, radial_unit, arc_length, init_gap).
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "cell_size": {"type": "Vector[2]"},
            "init_gap": {"type": "float"},
        }


@register_symbol
class RadialRepeatBricked(PartitionGrid):
    """Radial partition with bricked (offset) rings.
    
    Maps to polar_repeat_bricked_grid(grid, radial_unit, angular_unit, init_gap).
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "cell_size": {"type": "Vector[2]"},
            "init_gap": {"type": "float"},
        }


@register_symbol
class RadialRepeatFixedArcBricked(PartitionGrid):
    """Radial partition with fixed arc length and bricked rings.
    
    Maps to polar_repeat_fixed_arc_bricked_grid(grid, radial_unit, arc_length, init_gap).
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "cell_size": {"type": "Vector[2]"},
            "init_gap": {"type": "float"},
        }


@register_symbol
class AngularRepeat(PartitionGrid):
    """Partition by angular sectors (pie slices)."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "angular_unit": {"type": "float"},
        }


# ---- Irregular / Voronoi partitions ----

@register_symbol
class VoronoiRepeat(PartitionGrid):
    """Voronoi-based irregular partitioning (size-wise centroid placement)."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "cell_size": {"type": "Vector[2]"},
            "noise_rate": {"type": "float"},
        }


@register_symbol
class VoronoiRepeatRadialDeformed(PartitionGrid):
    """Voronoi partitioning with radial deformation of centroids."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "cell_size": {"type": "Vector[2]"},
            "noise_rate": {"type": "float"},
        }


@register_symbol
class IrregularRectRepeat(PartitionGrid):
    """Aperiodic rectangular partitioning with noise.
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
class DelaunayRepeat(PartitionGrid):
    """Delaunay triangulation-based partitioning."""
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "cell_size": {"type": "Vector[2]"},
        }


@register_symbol
class SDFRingPartition(PartitionGrid):
    """Partition space into concentric SDF contour rings around a shape.

    Used for RSP shape_sdf and shape_overlay splits.
    Evaluates shape_expr as SDF on grid, rounds by sdf_step_size to get ring IDs.
    """
    @classmethod
    def default_spec(cls):
        return {
            "grid": {"type": "Expr"},
            "shape_expr": {"type": "Expr"},
            "sdf_step_size": {"type": "float"},
            "ring_counts": {"type": "int"},
            "growth_mode": {"type": 'Enum["linear"|"logarithmic"]', "optional": True},
        }

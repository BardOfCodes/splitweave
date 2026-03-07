
from yacs.config import CfgNode as CN
import numpy as np
import torch as th
import geolipi.symbolic as gls
from .mtp_tile import sample_retro_tile, default_retro_tile
from .utils import add_to_registry

region_types = [
    'x_strip', 'y_strip', 'polar_r', 'polar_a',
    'diamond_board', "checkerboard" , "hex_strip",
    'ring_hex', 'ray_hex', "ray_diamond",
    'hex_scatter', "diamond_scatter",
    "polar_checkerboard",
    "shape_sdf", "shape_overlay",
    "voronoi_checkerboard", "irregular_rect_checkerboard"
    'polar_fixed_arc', 'polar_fixed_arc_bricked',
    # voronoi_radial, irregular_radial
    # radial_triangular
    # gs.RadialVoronoiRepeat, # Not really useful
    # gs.RadialIrregularRectRepeat, # Not really useful
    # K Tile Overlay
    # "k Geom " # sample a base shape, eval and use sdf contours
    # 
]

GRID_SIZE_MIN = 0.05
GRID_SIZE_MAX = 0.3
DIAMOND_GRID_SIZE_MIN = 0.2
DIAMOND_GRID_SIZE_MAX = 0.4
HEXAGONAL_GRID_SIZE_MIN = 0.1
HEXAGONAL_GRID_SIZE_MAX = 0.3
SIMPLE_POLAR_RAD_MIN = 0.15
SIMPLE_POLAR_RAD_MAX = 0.3
SIMPLE_POLAR_ANG_MIN = np.pi/12
SIMPLE_POLAR_ANG_MAX = np.pi/9
HEXAGONAL_GRID_SIZE_MIN = 0.15
HEXAGONAL_GRID_SIZE_MAX = 0.3
DIAMOND_MIN_COUNT = 3
DIAMOND_MAX_COUNT = 7

VN_C_MIN = 0.1
VN_C_MAX = 0.3
VN_N_MIN = 0.5
VN_N_MAX = 0.9

IN_X_MIN = 0.1
IN_X_MAX = 0.3
IN_N_MIN = 0.5
IN_N_MAX = 0.8

deform_rate = 0.80
match_xy_likelihood = 0.25

split_default_mapper, split_sampler_mapper = {}, {}

#### X Strip ####

@add_to_registry(split_sampler_mapper, "x_strip")
def sample_rect_repeat(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "x_strip"
    stripe_size = np.random.uniform(GRID_SIZE_MIN, GRID_SIZE_MAX)
    cfg.stripe_size = float(stripe_size)
    # Normalization
    return cfg

@add_to_registry(split_default_mapper, "x_strip")
def default_rect_repeat(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "x_strip"
    cfg.stripe_size = 0.15
    return cfg

#### Y Strip ####
@add_to_registry(split_sampler_mapper, "y_strip")
def sample_y_strip(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "y_strip"
    stripe_size = np.random.uniform(GRID_SIZE_MIN, GRID_SIZE_MAX)
    cfg.stripe_size = float(stripe_size)
    return cfg

@add_to_registry(split_default_mapper, "y_strip")
def default_y_strip(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "y_strip"
    cfg.stripe_size = 0.15
    return cfg

#### checkerboard ####
@add_to_registry(split_sampler_mapper, "checkerboard")
def sample_checkerboard(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "checkerboard"
    x_size = np.random.uniform(GRID_SIZE_MIN, GRID_SIZE_MAX) * 2
    cfg.x_size = float(x_size)
    y_size = np.random.uniform(GRID_SIZE_MIN, GRID_SIZE_MAX) * 2
    cfg.y_size = float(y_size)
    return cfg

@add_to_registry(split_default_mapper, "checkerboard")
def default_checkerboard(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "checkerboard"
    cfg.x_size = 0.3
    cfg.y_size = 0.3
    return cfg

#### diamond_board ####
@add_to_registry(split_sampler_mapper, "diamond_board")
def sample_diamond_board(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "diamond_board"
    stripe_size = np.random.uniform(DIAMOND_GRID_SIZE_MIN, DIAMOND_GRID_SIZE_MAX)
    cfg.stripe_size = float(stripe_size)
    return cfg

@add_to_registry(split_default_mapper, "diamond_board")
def default_diamond_board(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "diamond_board"
    cfg.stripe_size = 0.3
    return cfg

#### polar_r ####
@add_to_registry(split_sampler_mapper, "polar_r")
def sample_polar_r(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "polar_r"
    radial_unit = np.random.uniform(SIMPLE_POLAR_RAD_MIN, SIMPLE_POLAR_RAD_MAX)
    cfg.radial_unit = float(radial_unit)
    return cfg

@add_to_registry(split_default_mapper, "polar_r")
def default_polar_r(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "polar_r"
    cfg.radial_unit = 0.2
    return cfg

#### polar_a ####
@add_to_registry(split_sampler_mapper, "polar_a")
def sample_polar_a(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "polar_a"
    angular_unit = np.random.uniform(SIMPLE_POLAR_ANG_MIN, SIMPLE_POLAR_ANG_MAX)
    cfg.angular_unit = float(angular_unit)
    return cfg

@add_to_registry(split_default_mapper, "polar_a")
def default_polar_a(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "polar_a"
    cfg.angular_unit = np.pi
    return cfg

#### hex_strip ####
@add_to_registry(split_sampler_mapper, "hex_strip")
def sample_hex_strip(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "hex_strip"
    stripe_size = np.random.uniform(HEXAGONAL_GRID_SIZE_MIN, HEXAGONAL_GRID_SIZE_MAX)
    cfg.stripe_size = float(stripe_size)
    return cfg

@add_to_registry(split_default_mapper, "hex_strip")
def default_hex_strip(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "hex_strip"
    cfg.stripe_size = 0.2
    return cfg

#### ring_hex ####
@add_to_registry(split_sampler_mapper, "ring_hex")
def sample_ring_hex(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "ring_hex"
    stripe_size = np.random.uniform(HEXAGONAL_GRID_SIZE_MIN, HEXAGONAL_GRID_SIZE_MAX)
    cfg.stripe_size = float(stripe_size)
    return cfg

@add_to_registry(split_default_mapper, "ring_hex")
def default_ring_hex(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "ring_hex"
    cfg.stripe_size = 0.2
    return cfg

#### ray_hex ####
@add_to_registry(split_sampler_mapper, "ray_hex")
def sample_ray_hex(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "ray_hex"
    k = np.random.randint(DIAMOND_MIN_COUNT, DIAMOND_MAX_COUNT)
    size = 1/ (2 * np.cos(np.pi/6) * k) 
    cfg.stripe_size = float(size)
    cfg.count = int(k)
    return cfg

@add_to_registry(split_default_mapper, "ray_hex")
def default_ray_hex(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "ray_hex"
    cfg.stripe_size = 0.2
    cfg.count = 5
    return cfg

#### ray_diamond ####
@add_to_registry(split_sampler_mapper, "ray_diamond")
def sample_ray_diamond(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "ray_diamond"
    k = np.random.randint(DIAMOND_MIN_COUNT, DIAMOND_MAX_COUNT)
    size = 1/ (2 * np.cos(np.pi/6) * k) 
    cfg.stripe_size = float(size)
    cfg.count = int(k)
    return cfg

@add_to_registry(split_default_mapper, "ray_diamond")
def default_ray_diamond(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "ray_diamond"
    cfg.stripe_size = 0.2
    cfg.count = 5
    return cfg

#### hex_scatter ####
@add_to_registry(split_sampler_mapper, "hex_scatter")
def sample_hex_scatter(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "hex_scatter"
    k = np.random.randint(DIAMOND_MIN_COUNT, DIAMOND_MAX_COUNT)
    size = 1/ (2 * np.cos(np.pi/6) * k) 
    cfg.stripe_size = float(size)
    cfg.count = int(k)
    return cfg

@add_to_registry(split_default_mapper, "hex_scatter")
def default_hex_scatter(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "hex_scatter"
    cfg.stripe_size = 0.2
    cfg.count = 5
    return cfg

#### diamond_scatter ####
@add_to_registry(split_sampler_mapper, "diamond_scatter")
def sample_diamond_scatter(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "diamond_scatter"
    k = np.random.randint(DIAMOND_MIN_COUNT, DIAMOND_MAX_COUNT)
    size = 1/ (2 * np.cos(np.pi/6) * k) 
    cfg.stripe_size = float(size)
    cfg.count = int(k)
    return cfg

@add_to_registry(split_default_mapper, "diamond_scatter")
def default_diamond_scatter(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "diamond_scatter"
    cfg.stripe_size = 0.2
    cfg.count
    return cfg

#### polar_checkerboard ####
@add_to_registry(split_sampler_mapper, "polar_checkerboard")
def sample_polar_checkerboard(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "polar_checkerboard"
    radial_unit = np.random.uniform(SIMPLE_POLAR_RAD_MIN, SIMPLE_POLAR_RAD_MAX)
    cfg.radial_unit = float(radial_unit)
    angular_unit = np.random.uniform(SIMPLE_POLAR_ANG_MIN, SIMPLE_POLAR_ANG_MAX)
    cfg.angular_unit = float(angular_unit)
    return cfg

@add_to_registry(split_default_mapper, "polar_checkerboard")
def default_polar_checkerboard(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "polar_checkerboard"
    cfg.radial_unit = 0.2
    cfg.angular_unit = np.pi/6
    return cfg

#### polar_fixed_arc ####
@add_to_registry(split_sampler_mapper, "polar_fixed_arc")
def sample_polar_fixed_arc(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "polar_fixed_arc"
    radial_unit = np.random.uniform(SIMPLE_POLAR_RAD_MIN, SIMPLE_POLAR_RAD_MAX)
    cfg.radial_unit = float(radial_unit)
    angular_unit = np.random.uniform(SIMPLE_POLAR_ANG_MIN, SIMPLE_POLAR_ANG_MAX)
    cfg.angular_unit = float(angular_unit)
    return cfg

@add_to_registry(split_default_mapper, "polar_fixed_arc")
def default_polar_fixed_arc(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "polar_fixed_arc"
    cfg.radial_unit = 0.2
    cfg.angular_unit = np.pi/6
    return

#### polar_fixed_arc_bricked ####
@add_to_registry(split_sampler_mapper, "polar_fixed_arc_bricked")
def sample_polar_fixed_arc_bricked(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "polar_fixed_arc_bricked"
    radial_unit = np.random.uniform(SIMPLE_POLAR_RAD_MIN, SIMPLE_POLAR_RAD_MAX)
    cfg.radial_unit = float(radial_unit)
    angular_unit = np.random.uniform(SIMPLE_POLAR_ANG_MIN, SIMPLE_POLAR_ANG_MAX)
    cfg.angular_unit = float(angular_unit)
    return cfg

@add_to_registry(split_default_mapper, "polar_fixed_arc_bricked")
def default_polar_fixed_arc_bricked(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "polar_fixed_arc_bricked"
    cfg.radial_unit = 0.2
    cfg.angular_unit = np.pi/6
    return cfg

#### voronoi_checkerboard ####
@add_to_registry(split_sampler_mapper, "voronoi_checkerboard")
def sample_voronoi_checkerboard(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "voronoi_checkerboard"
    x_size = np.random.uniform(VN_C_MIN, VN_C_MAX)
    noise_rate = np.random.uniform(VN_N_MIN, VN_N_MAX)
    match_xy = np.random.choice([True, False], p=[match_xy_likelihood, 1-match_xy_likelihood])
    if match_xy:
        y_size = x_size
    else:
        y_size = np.random.uniform(VN_C_MIN, VN_C_MAX)

    cfg.x_size = float(x_size)
    cfg.y_size = float(y_size)
    seed = np.random.randint(0, 100000)
    cfg.seed = int(seed)
    cfg.noise_rate = float(noise_rate)
    radial = np.random.choice([True, False])
    cfg.radial = bool(radial)
    return cfg

@add_to_registry(split_default_mapper, "voronoi_checkerboard")
def default_voronoi_checkerboard(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "voronoi_checkerboard"
    cfg.x_size = 0.2
    cfg.y_size = 0.2
    cfg.noise_rate = 0.7
    seed = np.random.randint(0, 100000)
    cfg.seed = int(seed)
    cfg.radial = False
    return cfg

#### irregular_rect_checkerboard ####
@add_to_registry(split_sampler_mapper, "irregular_rect_checkerboard")
def sample_irregular_rect_checkerboard(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "irregular_rect_checkerboard"
    x_size = np.random.uniform(IN_X_MIN, IN_X_MAX)
    noise_rate = np.random.uniform(IN_N_MIN, IN_N_MAX)
    match_xy = np.random.choice([True, False], p=[match_xy_likelihood, 1-match_xy_likelihood])
    if match_xy:
        y_size = x_size
    else:
        y_size = np.random.uniform(IN_X_MIN, IN_X_MAX)

    cfg.x_size = float(x_size)
    cfg.y_size = float(y_size)
    cfg.noise_rate = float(noise_rate)

    seed = np.random.randint(0, 100000)
    cfg.seed = int(seed)
    radial = np.random.choice([True, False])
    cfg.radial = bool(radial)
    return cfg

@add_to_registry(split_default_mapper, "irregular_rect_checkerboard")
def default_irregular_rect_checkerboard(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "irregular_rect_checkerboard"
    cfg.x_size = 0.2
    cfg.y_size = 0.2
    cfg.noise_rate = 0.7

    seed = np.random.randint(0, 100000)
    cfg.seed = int(seed)
    cfg.radial = False
    return cfg

#### shape_sdf ####
@add_to_registry(split_sampler_mapper, "shape_sdf")
def sample_shape_sdf(parent_cfg, all_filenames, filename_to_indices, *args, **kwargs):
    cfg = CN()
    cfg._type = "shape_sdf"
    cfg.tilespec = sample_retro_tile(all_filenames, filename_to_indices)
    step_size = np.random.uniform(0.2, 0.4) * 0.2
    cfg.sdf_step_size = float(step_size)
    shape_scale = np.random.uniform(0.2, 0.4)
    cfg.shape_scale = float(shape_scale)
    growth_mode = np.random.choice(["linear", "logarithmic"])
    cfg.growth_mode = str(growth_mode)
    return cfg

@add_to_registry(split_default_mapper, "shape_sdf")
def default_shape_sdf(parent_cfg, all_filenames, filename_to_indices, *args, **kwargs):
    cfg = CN()
    cfg._type = "shape_sdf"
    cfg.tilespec = default_retro_tile(all_filenames, filename_to_indices)
    cfg.sdf_step_size = 0.3
    cfg.shape_scale = 0.2
    cfg.growth_mode = "linear"
    return cfg

#### shape_overlay ####
@add_to_registry(split_sampler_mapper, "shape_overlay")
def sample_shape_overlay(parent_cfg, all_filenames, filename_to_indices, *args, **kwargs):
    cfg = CN()
    cfg._type = "shape_overlay"
    cfg.tilespec = sample_retro_tile(all_filenames, filename_to_indices)
    step_size = np.random.uniform(0.05, 0.1)
    cfg.sdf_step_size = float(step_size)
    shape_scale = np.random.uniform(2, 3)
    cfg.shape_scale = float(shape_scale)
    growth_mode = np.random.choice(["linear", "logarithmic"])
    cfg.growth_mode = str(growth_mode)
    return cfg

@add_to_registry(split_default_mapper, "shape_overlay")
def default_shape_overlay(parent_cfg, all_filenames, filename_to_indices, *args, **kwargs):
    cfg = CN()
    cfg._type = "shape_overlay"
    cfg.tilespec = default_retro_tile(all_filenames, filename_to_indices)
    cfg.sdf_step_size = 0.1
    cfg.shape_scale = 2.5
    cfg.growth_mode = "linear"
    return cfg

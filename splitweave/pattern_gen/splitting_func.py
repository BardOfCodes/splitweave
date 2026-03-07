from yacs.config import CfgNode as CN
import numpy as np
import torch as th
from .utils import add_to_registry

X_SIZE_MIN = 0.15
X_SIZE_MAX = 0.35
HEXAGONAL_GRID_SIZE_MIN = 0.15
HEXAGONAL_GRID_SIZE_MAX = 0.3
# Overlap
OVERLAP_SINGLE_MODES = ['l', 'c', "cc", 'r']
OVERLAP_DOUBLE_MODES = ['axial', 'diagonal']
# POLAR
SIMPLE_POLAR_RAD_MIN = 0.15
SIMPLE_POLAR_RAD_MAX = 0.3
SIMPLE_POLAR_ANG_MIN = np.pi/6
SIMPLE_POLAR_ANG_MAX = np.pi/3
FIXED_ARC_RAD_MIN = 0.15
FIXED_ARC_RAD_MAX = 0.3
FIXED_ARC_ANG_MIN = 0.2
FIXED_ARC_ANG_MAX = 0.5
# Irregular
VR_N_MIN = 0.3
VR_N_MAX = 0.9
IN_N_MIN = 0.3
IN_N_MAX = 0.9
FILL_CUTOFF_MIN = 0.05
FILL_CUTOFF_MAX = 0.1

split_sampler_mapper = {}
split_variable_mapper = {}
split_default_mapper = {}


############ RectRepeat ############

@add_to_registry(split_sampler_mapper, "RectRepeat")
def sample_rect_repeat(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "RectRepeat"
    x_size = np.random.uniform(X_SIZE_MIN, X_SIZE_MAX)
    y_size = np.random.uniform(X_SIZE_MIN, X_SIZE_MAX)
    cfg.x_size = float(x_size)
    cfg.y_size = float(y_size)
    # Normalization
    normalization_mode = np.random.choice([0, 1])
    cfg.normalization_mode = int(normalization_mode)
    split_size = min(x_size, y_size)
    return cfg, split_size

@add_to_registry(split_variable_mapper, "RectRepeat")
def var_resampler_rect_repeat(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_x_size():
        x_size = float(np.random.uniform(X_SIZE_MIN, X_SIZE_MAX))
        def edit_func(cfg):
            cfg.x_size = x_size
            return cfg
        return edit_func
    def replace_y_size():
        y_size = float(np.random.uniform(X_SIZE_MIN, X_SIZE_MAX))
        def edit_func(cfg):
            cfg.y_size = y_size
            return cfg
        return edit_func
    resamplers["x_size"] = replace_x_size
    resamplers["y_size"] = replace_y_size
    # if we don't want to make analogies with a variable, just have a none func
    resamplers["normalization_mode"] = None

    return resamplers

@add_to_registry(split_default_mapper, "RectRepeat")
def default_rect_repeat(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "RectRepeat"
    cfg.x_size = 0.2
    cfg.y_size = 0.2
    cfg.normalization_mode = 0
    return cfg

############ RectRepeatFitting ############

@add_to_registry(split_sampler_mapper, "RectRepeatFitting")
def sample_rect_repeat_fitting(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "RectRepeatFitting"
    tile_ratio = kwargs.get("tile_ratio", None)
    # The other way to access it would be with the parent_cfg
    if tile_ratio is None:
        tile_ratio = (1., 1.)
    x_size = np.random.uniform(X_SIZE_MIN, X_SIZE_MAX)

    if tile_ratio[0] == 1:
        y_size = x_size * tile_ratio[1]
    elif tile_ratio[2] == 1:
        y_size = x_size / tile_ratio[0] 
    else:
        print(f"Invalid tile ratio: {tile_ratio}")
        raise ValueError("Invalid tile ratio")
    cfg.fit_x_size = float(x_size) # different from parent.
    cfg.fit_y_size = float(y_size) # No for editing.
    # Only one normalization mode
    cfg.tile_ratio = tile_ratio
    normalization_mode = np.random.choice([0, 1])
    cfg.normalization_mode = int(normalization_mode)
    split_size = min(x_size, y_size)
    return cfg, split_size


@add_to_registry(split_variable_mapper, "RectRepeatFitting")
def var_resampler_rect_repeat_fitting(current_cfg, *args, **kwargs):
    resamplers = {}
    # fit x size:
    def replace_fit_x_size():
        fit_x_size = float(np.random.uniform(X_SIZE_MIN, X_SIZE_MAX))
        def edit_func(cfg):
            older_ratio = cfg.fit_x_size / cfg.fit_y_size
            cfg.fit_x_size = fit_x_size
            fit_y_size = fit_x_size / older_ratio
            cfg.fit_y_size = fit_y_size
            return cfg
        return edit_func
    resamplers["fit_x_size"] = replace_fit_x_size
    resamplers["fit_y_size"] = None
    resamplers["tile_ratio"] = None
    resamplers["normalization_mode"] = None
    return resamplers
    
@add_to_registry(split_default_mapper, "RectRepeatFitting")
def default_rect_repeat_fitting(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "RectRepeatFitting"
    cfg.x_size = 0.25

    if tile_ratio is None:
        tile_ratio = (1., 1.)
    x_size = np.random.uniform(X_SIZE_MIN, X_SIZE_MAX)

    if tile_ratio[0] == 1:
        y_size = x_size * tile_ratio[1]
    elif tile_ratio[2] == 1:
        y_size = x_size / tile_ratio[0] 
    else:
        print(f"Invalid tile ratio: {tile_ratio}")
        raise ValueError("Invalid tile ratio")
    cfg.fit_x_size = float(x_size)
    cfg.fit_y_size = float(y_size)
    cfg.tile_ratio = tile_ratio
    normalization_mode = np.random.choice([0, 1])
    cfg.normalization_mode = int(normalization_mode)

    return cfg

############ OverlapX ############
@add_to_registry(split_sampler_mapper, "OverlapX")
def sample_overlap_x(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "OverlapX"
    o_x_x_size = np.random.uniform(X_SIZE_MIN, X_SIZE_MAX)
    o_x_y_size = np.random.uniform(X_SIZE_MIN, o_x_x_size)
    o_x_d = np.random.uniform(1, 2)
    # modes
    single_mode = np.random.choice(OVERLAP_SINGLE_MODES)
    ref_horiz = np.random.choice([True, False])
    ref_vertical = np.random.choice([True, False])
    multi_tile = np.random.choice([True, False])
    cfg.o_x_x_size = float(o_x_x_size)
    cfg.o_x_y_size = float(o_x_y_size)
    cfg.o_x_inter_tile_dist = float(o_x_d)
    cfg.single_mode = str(single_mode)
    cfg.ref_horiz = bool(ref_horiz)
    cfg.ref_vertical = bool(ref_vertical)
    cfg.multi_tile = bool(multi_tile)
    cfg.normalization_mode = 0
    split_size = min(o_x_x_size, o_x_y_size)
    return cfg, split_size

@add_to_registry(split_variable_mapper, "OverlapX")
def var_resample_overlap_x(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_o_x_x_size():
        o_x_x_size = float(np.random.uniform(X_SIZE_MIN, X_SIZE_MAX))
        def edit_func(cfg):
            cfg.o_x_x_size = o_x_x_size
            if cfg.o_x_y_size > o_x_x_size:
                cfg.o_x_y_size = o_x_x_size
            return cfg
        return edit_func
    def replace_o_x_y_size():
        o_x_y_size = float(np.random.uniform(X_SIZE_MIN, current_cfg.o_x_x_size))
        def edit_func(cfg, o_x_y_size=o_x_y_size):
            current_o_x_x_size = cfg.o_x_x_size
            if o_x_y_size > current_o_x_x_size:
                o_x_y_size = current_o_x_x_size
            cfg.o_x_y_size = o_x_y_size
            return cfg
        return edit_func
    def replace_o_x_inter_tile_dist():
        o_x_d = float(np.random.uniform(1, 2))
        def edit_func(cfg):
            cfg.o_x_inter_tile_dist = o_x_d
            return cfg
        return edit_func
    def replace_single_mode():
        single_mode = str(np.random.choice(OVERLAP_SINGLE_MODES))
        def edit_func(cfg):
            cfg.single_mode = single_mode
            return cfg
        return edit_func
    def replace_ref_horiz():
        ref_horiz = bool(np.random.choice([True, False]))
        def edit_func(cfg):
            cfg.ref_horiz = ref_horiz
            return cfg
        return edit_func
    def replace_multi_tile():
        multi_tile = False # np.random.choice([True, False])
        #  CAN Only Switch "OFF"

        def edit_func(cfg):
            cfg.multi_tile = multi_tile
            return cfg
        return edit_func
    
    resamplers["o_x_x_size"] = replace_o_x_x_size
    resamplers["o_x_y_size"] = replace_o_x_y_size
    resamplers["o_x_inter_tile_dist"] = replace_o_x_inter_tile_dist
    resamplers["single_mode"] = replace_single_mode
    resamplers["ref_horiz"] = replace_ref_horiz
    resamplers["multi_tile"] = replace_multi_tile
    resamplers["normalization_mode"] = None
    return resamplers

@add_to_registry(split_default_mapper, "OverlapX")
def default_overlap_x(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "OverlapX"
    cfg.o_x_x_size = 0.25

    tile_ratio = kwargs.get("tile_ratio", None)
    if tile_ratio is None:
        tile_ratio = (1., 1.)

    if tile_ratio[0] == 1:
        y_size = cfg.o_x_x_size * tile_ratio[1]
    elif tile_ratio[2] == 1:
        y_size = cfg.o_x_x_size / tile_ratio[0] 
    else:
        print(f"Invalid tile ratio: {tile_ratio}")
    cfg.o_x_y_size = y_size
    cfg.o_x_inter_tile_dist = 1.0
    cfg.single_mode = "c"
    cfg.ref_horiz = False
    cfg.ref_vertical = False
    cfg.multi_tile = False
    cfg.normalization_mode = 0
    return cfg

############ OverlapY ############

@add_to_registry(split_sampler_mapper, "OverlapY")
def sample_overlap_y(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "OverlapY"
    o_y_y_size = np.random.uniform(X_SIZE_MIN, X_SIZE_MAX)
    o_y_x_size = np.random.uniform(X_SIZE_MIN, o_y_y_size)
    o_y_d = np.random.uniform(1, 2)
    # modes
    single_mode = np.random.choice(OVERLAP_SINGLE_MODES)
    ref_horiz = np.random.choice([True, False])
    ref_vertical = np.random.choice([True, False])
    multi_tile = np.random.choice([True, False])
    cfg.o_y_y_size = float(o_y_y_size)
    cfg.o_y_x_size = float(o_y_x_size)
    cfg.o_y_inter_tile_dist = float(o_y_d)
    cfg.single_mode = str(single_mode)
    cfg.ref_horiz = bool(ref_horiz)
    cfg.ref_vertical = bool(ref_vertical)
    cfg.multi_tile = bool(multi_tile)
    cfg.normalization_mode = 0
    split_size = min(o_y_y_size, o_y_x_size)
    return cfg, split_size

@add_to_registry(split_variable_mapper, "OverlapY")
def var_resample_overlap_y(current_cfg, *args, **kwargs):
    sampler = {}
    def replace_o_y_y_size():
        o_y_y_size = float(np.random.uniform(X_SIZE_MIN, X_SIZE_MAX))
        def edit_func(cfg):
            cfg.o_y_y_size = o_y_y_size
            if cfg.o_y_x_size > o_y_y_size:
                cfg.o_y_x_size = o_y_y_size
            return cfg
        return edit_func
    def replace_o_y_x_size():
        o_y_x_size = float(np.random.uniform(X_SIZE_MIN, current_cfg.o_y_y_size))
        def edit_func(cfg, o_y_x_size=o_y_x_size):
            current_o_y_y_size = cfg.o_y_y_size
            if o_y_x_size > current_o_y_y_size:
                o_y_x_size = current_o_y_y_size
            cfg.o_y_x_size = o_y_x_size
            return cfg
        return edit_func
    def replace_o_y_inter_tile_dist():
        o_y_d = float(np.random.uniform(1, 2))
        def edit_func(cfg):
            cfg.o_y_inter_tile_dist = o_y_d
            return cfg
        return edit_func
    def replace_single_mode():
        single_mode = str(np.random.choice(OVERLAP_SINGLE_MODES))
        def edit_func(cfg):
            cfg.single_mode = single_mode
            return cfg
        return edit_func
    def replace_ref_vertical():
        ref_vertical = bool(np.random.choice([True, False]))
        def edit_func(cfg):
            cfg.ref_vertical = ref_vertical
            return cfg
        return edit_func
    def replace_multi_tile():
        multi_tile = False # np.random.choice([True, False])
        def edit_func(cfg):
            cfg.multi_tile = multi_tile
            return cfg
        return edit_func
    sampler["o_y_y_size"] = replace_o_y_y_size
    sampler["o_y_x_size"] = replace_o_y_x_size
    sampler["o_y_inter_tile_dist"] = replace_o_y_inter_tile_dist
    sampler["single_mode"] = replace_single_mode
    sampler["ref_vertical"] = replace_ref_vertical
    sampler["multi_tile"] = replace_multi_tile
    sampler["normalization_mode"] = None
    return sampler

@add_to_registry(split_default_mapper, "OverlapY")
def default_overlap_y(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "OverlapY"
    cfg.o_y_y_size = 0.25

    tile_ratio = kwargs.get("tile_ratio", None)
    if tile_ratio is None:
        tile_ratio = (1., 1.)

    if tile_ratio[0] == 1:
        x_size = cfg.o_y_y_size / tile_ratio[1]
    elif tile_ratio[2] == 1:
        x_size = cfg.o_y_y_size * tile_ratio[0] 
    else:
        print(f"Invalid tile ratio: {tile_ratio}")

    cfg.o_y_x_size = x_size
    cfg.o_y_inter_tile_dist = 1.0
    cfg.single_mode = "c"
    cfg.ref_vertical = False
    cfg.multi_tile = False
    cfg.normalization_mode = 0
    return cfg

### OverlapXY ###

@add_to_registry(split_sampler_mapper, "OverlapXY")
def sample_overlap_xy(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "OverlapXY"
    o_xy_grid_size = np.random.uniform(X_SIZE_MAX, X_SIZE_MAX)

    o_xy_d = np.random.uniform(1, 2)
    # modes
    x_overlap_mode = np.random.choice(OVERLAP_SINGLE_MODES)
    # main_mode 
    main_mode = np.random.choice(OVERLAP_DOUBLE_MODES)

    ref_horiz = np.random.choice([True, False])
    ref_vert = np.random.choice([True, False])
    multi_tile = np.random.choice([True, False])

    cfg.o_xy_grid_size = float(o_xy_grid_size)
    cfg.o_xy_inter_tile_dist = float(o_xy_d)
    cfg.overlap_mode = str(x_overlap_mode)
    cfg.main_mode = str(main_mode)
    cfg.ref_horiz = bool(ref_horiz)
    cfg.ref_vertical = bool(ref_vert)

    cfg.multi_tile = bool(multi_tile)
    cfg.normalization_mode = 0
    split_size = o_xy_grid_size
    return cfg, split_size

@add_to_registry(split_variable_mapper, "OverlapXY")
def var_resample_overlap_xy(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_o_xy_grid_size():
        o_xy_grid_size = float(np.random.uniform(X_SIZE_MIN, X_SIZE_MAX))
        def edit_func(cfg):
            cfg.o_xy_grid_size = o_xy_grid_size
            return cfg
        return edit_func
    def replace_o_xy_inter_tile_dist():
        o_xy_d = float(np.random.uniform(1, 2))
        def edit_func(cfg):
            cfg.o_xy_inter_tile_dist = o_xy_d
            return cfg
        return edit_func
    def replace_x_overlap_mode():
        x_overlap_mode = str(np.random.choice(OVERLAP_SINGLE_MODES))
        def edit_func(cfg):
            cfg.x_overlap_mode = x_overlap_mode
            return cfg
        return edit_func
    def replace_main_mode():
        main_mode = str(np.random.choice(OVERLAP_DOUBLE_MODES))
        def edit_func(cfg):
            cfg.main_mode = main_mode
            return cfg
        return edit_func
    def replace_ref():
        ref = bool(np.random.choice([True, False]))
        def edit_func(cfg):
            cfg.ref = ref
            return cfg
        return edit_func
    def replace_multi_tile():
        multi_tile = False # np.random.choice([True, False])
        def edit_func(cfg):
            cfg.multi_tile = multi_tile
            return cfg
        return edit_func
    resamplers["o_xy_grid_size"] = replace_o_xy_grid_size
    resamplers["o_xy_inter_tile_dist"] = replace_o_xy_inter_tile_dist
    resamplers["overlap_mode"] = replace_x_overlap_mode
    resamplers["main_mode"] = replace_main_mode
    resamplers["ref"] = replace_ref
    resamplers["multi_tile"] = replace_multi_tile
    resamplers["normalization_mode"] = None
    return resamplers

@add_to_registry(split_default_mapper, "OverlapXY")
def default_overlap_xy(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "OverlapXY"
    cfg.o_xy_grid_size = 0.25
    cfg.o_xy_inter_tile_dist = 1.0
    cfg.overlap_mode = "c"
    cfg.main_mode = "axial"
    cfg.ref = False
    cfg.multi_tile = False
    cfg.normalization_mode = 0
    return cfg

############ RectRepeatShiftedX ############

@add_to_registry(split_sampler_mapper, "RectRepeatShiftedX")
def sample_rect_repeat_shifted_x(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "RectRepeatShiftedX"
    x_size = np.random.uniform(X_SIZE_MIN, X_SIZE_MAX)
    y_size = np.random.uniform(X_SIZE_MIN, X_SIZE_MAX)
    if np.random.choice([True, False]):
        x_shift = x_size / 2
    else:
        x_shift = np.random.uniform(0, x_size)
    cfg.x_size = float(x_size)
    cfg.y_size = float(y_size)
    cfg.x_shift = float(x_shift)
    normalization_mode = np.random.choice([0, 1])
    cfg.normalization_mode = int(normalization_mode)
    split_size = min(x_size, y_size)
    return cfg, split_size

@add_to_registry(split_variable_mapper, "RectRepeatShiftedX")
def var_resample_rect_repeat_shifted_x(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_x_size():
        x_size = float(np.random.uniform(X_SIZE_MIN, X_SIZE_MAX))
        def edit_func(cfg):
            cfg.x_size = x_size
            return cfg
        return edit_func
    def replace_y_size():
        y_size = float(np.random.uniform(X_SIZE_MIN, X_SIZE_MAX))
        def edit_func(cfg):
            cfg.y_size = y_size
            return cfg
        return edit_func
    def replace_x_shift():
        set_half = bool(np.random.choice([True, False]))
        x_shift = float(np.random.uniform(0, current_cfg.x_size))
        def edit_func(cfg):
            if set_half:
                x_shift = cfg.x_size / 2
            else:
                if x_shift > cfg.x_size:
                    x_shift = cfg.x_size * 0.9
            cfg.x_shift = x_shift
            return cfg
        return edit_func
    resamplers["x_size"] = replace_x_size
    resamplers["y_size"] = replace_y_size
    resamplers["x_shift"] = replace_x_shift
    resamplers["normalization_mode"] = None
    return resamplers

@add_to_registry(split_default_mapper, "RectRepeatShiftedX")
def default_rect_repeat_shifted_x(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "RectRepeatShiftedX"
    cfg.x_size = 0.25
    cfg.y_size = 0.25
    cfg.x_shift = 0.125
    cfg.normalization_mode = 0
    return cfg

############ RectRepeatShiftedY ############
@add_to_registry(split_sampler_mapper, "RectRepeatShiftedY")
def sample_rect_repeat_shifted_y(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "RectRepeatShiftedY"
    x_size = np.random.uniform(X_SIZE_MIN, X_SIZE_MAX)
    y_size = np.random.uniform(X_SIZE_MIN, X_SIZE_MAX)
    if np.random.choice([True, False]):
        y_shift = y_size / 2
    else:
        y_shift = np.random.uniform(0, y_size)
    cfg.x_size = float(x_size)
    cfg.y_size = float(y_size)
    cfg.y_shift = float(y_shift)
    normalization_mode = np.random.choice([0, 1])
    cfg.normalization_mode = int(normalization_mode)
    split_size = min(x_size, y_size)
    return cfg, split_size

@add_to_registry(split_variable_mapper, "RectRepeatShiftedY")
def var_resample_rect_repeat_shifted_y(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_x_size():
        x_size = float(np.random.uniform(X_SIZE_MIN, X_SIZE_MAX))
        def edit_func(cfg):
            cfg.x_size = x_size
            return cfg
        return edit_func
    def replace_y_size():
        y_size = float(np.random.uniform(X_SIZE_MIN, X_SIZE_MAX))
        def edit_func(cfg):
            cfg.y_size = y_size
            return cfg
        return edit_func
    def replace_y_shift():
        set_half = bool(np.random.choice([True, False]))
        y_shift = float(np.random.uniform(0, current_cfg.y_size))
        def edit_func(cfg):
            if set_half:
                y_shift = cfg.y_size / 2
            else:
                if y_shift > cfg.y_size:
                    y_shift = cfg.y_size * 0.9
            cfg.y_shift = y_shift
            return cfg
        return edit_func
    resamplers["x_size"] = replace_x_size
    resamplers["y_size"] = replace_y_size
    resamplers["y_shift"] = replace_y_shift
    resamplers["normalization_mode"] = None
    return resamplers

@add_to_registry(split_default_mapper, "RectRepeatShiftedY")
def default_rect_repeat_shifted_y(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "RectRepeatShiftedY"
    cfg.x_size = 0.25
    cfg.y_size = 0.25
    cfg.y_shift = 0.125
    cfg.normalization_mode = 0
    return cfg

############ HexRepeat ############
@add_to_registry(split_sampler_mapper, "HexRepeat")
def sample_hex_repeat(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "HexRepeat"
    grid_size = np.random.uniform(HEXAGONAL_GRID_SIZE_MIN, HEXAGONAL_GRID_SIZE_MAX)
    cfg.grid_size = float(grid_size)
    normalization_mode = np.random.choice([0, 1])
    cfg.normalization_mode = int(normalization_mode)
    split_size = grid_size
    return cfg, split_size

@add_to_registry(split_variable_mapper, "HexRepeat")
def var_resample_hex_repeat(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_grid_size():
        grid_size = float(np.random.uniform(HEXAGONAL_GRID_SIZE_MIN, HEXAGONAL_GRID_SIZE_MAX))
        def edit_func(cfg):
            cfg.grid_size = grid_size
            return cfg
        return edit_func
    resamplers["grid_size"] = replace_grid_size
    resamplers["normalization_mode"] = None
    return resamplers

@add_to_registry(split_default_mapper, "HexRepeat")
def default_hex_repeat(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "HexRepeat"
    cfg.grid_size = 0.2
    cfg.normalization_mode = 0
    return cfg

############ HexRepeatY ############
@add_to_registry(split_sampler_mapper, "HexRepeatY")
def sample_hex_repeat_y(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "HexRepeatY"
    grid_size = np.random.uniform(HEXAGONAL_GRID_SIZE_MIN, HEXAGONAL_GRID_SIZE_MAX)
    cfg.grid_size = float(grid_size)
    normalization_mode = np.random.choice([0, 1])
    cfg.normalization_mode = int(normalization_mode)
    split_size = grid_size
    return cfg, split_size

@add_to_registry(split_variable_mapper, "HexRepeatY")
def var_resample_hex_repeat_y(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_grid_size():
        grid_size = float(np.random.uniform(HEXAGONAL_GRID_SIZE_MIN, HEXAGONAL_GRID_SIZE_MAX))
        def edit_func(cfg):
            cfg.grid_size = grid_size
            return cfg
        return edit_func
    resamplers["grid_size"] = replace_grid_size
    resamplers["normalization_mode"] = None
    return resamplers

@add_to_registry(split_default_mapper, "HexRepeatY")
def default_hex_repeat_y(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "HexRepeatY"
    cfg.grid_size = 0.2
    cfg.normalization_mode = 0
    return cfg


############ RadialRepeat ############
@add_to_registry(split_sampler_mapper, "RadialRepeat")
def sample_radial_repeat(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "RadialRepeat"
    radial_unit = np.random.uniform(SIMPLE_POLAR_RAD_MIN, SIMPLE_POLAR_RAD_MAX)
    angular_k = np.random.choice([4, 5, 6, 7, 8])
    center_x = np.random.uniform(-1.0, 1.0)
    center_y = np.random.uniform(-1.0, 1.0)
    angular_unit = np.pi / angular_k
    cfg.radial_unit = float(radial_unit)
    cfg.angular_unit = float(angular_unit)
    cfg.center_x = float(center_x)
    cfg.center_y = float(center_y)
    normalization_mode = np.random.choice([0, 1, 2])
    cfg.normalization_mode = int(normalization_mode)
    split_size = radial_unit
    return cfg, split_size

@add_to_registry(split_variable_mapper, "RadialRepeat")
def var_resample_radial_repeat(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_radial_unit():
        radial_unit = float(np.random.uniform(SIMPLE_POLAR_RAD_MIN, SIMPLE_POLAR_RAD_MAX))
        def edit_func(cfg):
            cfg.radial_unit = radial_unit
            return cfg
        return edit_func
    def replace_center():
        center_x = float(np.random.uniform(-1.0, 1.0))
        center_y = float(np.random.uniform(-1.0, 1.0))
        def edit_func(cfg):
            cfg.center_x = center_x
            cfg.center_y = center_y
            return cfg
        return edit_func
    
    def replace_angular_unit():
        angular_k = int(np.random.choice([4, 5, 6, 7, 8]))
        angular_unit = np.pi / angular_k
        def edit_func(cfg):
            cfg.angular_unit = angular_unit
            return cfg
        return edit_func
    resamplers["radial_unit"] = replace_radial_unit
    resamplers["angular_unit"] = replace_angular_unit
    resamplers["center_x"] = replace_center
    resamplers["center_y"] = replace_center
    resamplers["normalization_mode"] = None
    return resamplers

@add_to_registry(split_default_mapper, "RadialRepeat")
def default_radial_repeat(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "RadialRepeat"
    cfg.radial_unit = 0.25
    cfg.angular_unit = np.pi / 6
    cfg.center_x = 0.0
    cfg.center_y = 0.0
    cfg.normalization_mode = 0

    return cfg

############ RadialRepeatBricked ############

@add_to_registry(split_sampler_mapper, "RadialRepeatBricked")
def sample_radial_repeat_bricked(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "RadialRepeatBricked"
    radial_unit = np.random.uniform(SIMPLE_POLAR_RAD_MIN, SIMPLE_POLAR_RAD_MAX)
    angular_k = np.random.choice([4, 5, 6, 7, 8])
    angular_unit = np.pi / angular_k
    center_x = np.random.uniform(-1.0, 1.0)
    center_y = np.random.uniform(-1.0, 1.0)
    cfg.radial_unit = float(radial_unit)
    cfg.angular_unit = float(angular_unit)
    cfg.center_x = float(center_x)
    cfg.center_y = float(center_y)
    normalization_mode = np.random.choice([0, 1, 2])
    cfg.normalization_mode = int(normalization_mode)
    split_size = radial_unit
    return cfg, split_size

@add_to_registry(split_variable_mapper, "RadialRepeatBricked")
def var_resample_radial_repeat_bricked(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_radial_unit():
        radial_unit = float(np.random.uniform(SIMPLE_POLAR_RAD_MIN, SIMPLE_POLAR_RAD_MAX))
        def edit_func(cfg):
            cfg.radial_unit = radial_unit
            return cfg
        return edit_func
    def replace_center():
        center_x = float(np.random.uniform(-1.0, 1.0))
        center_y = float(np.random.uniform(-1.0, 1.0))
        def edit_func(cfg):
            cfg.center_x = center_x
            cfg.center_y = center_y
            return cfg
        return edit_func
    def replace_angular_unit():
        angular_unit = int(np.random.choice([4, 5, 6, 7, 8]))
        angular_unit = np.pi / angular_unit
        def edit_func(cfg):
            cfg.angular_unit = angular_unit
            return cfg
        return edit_func
    resamplers["radial_unit"] = replace_radial_unit
    resamplers["angular_unit"] = replace_angular_unit
    resamplers["center_x"] = replace_center
    resamplers["center_y"] = replace_center
    resamplers["normalization_mode"] = None
    return resamplers

@add_to_registry(split_default_mapper, "RadialRepeatBricked")
def default_radial_repeat_bricked(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "RadialRepeatBricked"
    cfg.radial_unit = 0.25
    cfg.angular_unit = np.pi / 6
    cfg.center_x = 0.0
    cfg.center_y = 0.0
    cfg.normalization_mode = 0
    return cfg

############ RadialRepeatFixedArc ############

@add_to_registry(split_sampler_mapper, "RadialRepeatFixedArc")
def sample_radial_repeat_fixed_arc(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "RadialRepeatFixedArc"
    radial_unit = np.random.uniform(FIXED_ARC_RAD_MIN, FIXED_ARC_RAD_MAX)
    angular_unit = np.random.uniform(FIXED_ARC_ANG_MIN, FIXED_ARC_ANG_MAX)
    normalization_mode = np.random.choice([0, 1, 2])
    center_x = np.random.uniform(-1.0, 1.0)
    center_y = np.random.uniform(-1.0, 1.0)
    cfg.radial_unit = float(radial_unit)
    cfg.arc_size = float(angular_unit)
    cfg.center_x = float(center_x)
    cfg.center_y = float(center_y)
    cfg.normalization_mode = int(normalization_mode)
    split_size = radial_unit
    return cfg, split_size

@add_to_registry(split_variable_mapper, "RadialRepeatFixedArc")
def var_resample_radial_repeat_fixed_arc(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_radial_unit():
        radial_unit = float(np.random.uniform(FIXED_ARC_RAD_MIN, FIXED_ARC_RAD_MAX))
        def edit_func(cfg):
            cfg.radial_unit = radial_unit
            return cfg
        return edit_func
    def replace_center():
        center_x = float(np.random.uniform(-1.0, 1.0))
        center_y = float(np.random.uniform(-1.0, 1.0))
        def edit_func(cfg):
            cfg.center_x = center_x
            cfg.center_y = center_y
            return cfg
        return edit_func
    def replace_angular_unit():
        angular_unit = float(np.random.uniform(FIXED_ARC_ANG_MIN, FIXED_ARC_ANG_MAX))
        def edit_func(cfg):
            cfg.arc_size = angular_unit
            return cfg
        return edit_func
    resamplers["radial_unit"] = replace_radial_unit
    resamplers["arc_size"] = replace_angular_unit
    resamplers["center_x"] = replace_center
    resamplers["center_y"] = replace_center
    resamplers["normalization_mode"] = None
    return resamplers

@add_to_registry(split_default_mapper, "RadialRepeatFixedArc")
def default_radial_repeat_fixed_arc(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "RadialRepeatFixedArc"
    cfg.radial_unit = 0.25
    cfg.arc_size = 0.35
    cfg.center_x = 0.0
    cfg.center_y = 0.0
    cfg.normalization_mode = 0
    return cfg

############ RadialRepeatFixedArcBricked ############

@add_to_registry(split_sampler_mapper, "RadialRepeatFixedArcBricked")
def sample_radial_repeat_fixed_arc_bricked(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "RadialRepeatFixedArcBricked"
    radial_unit = np.random.uniform(FIXED_ARC_RAD_MIN, FIXED_ARC_RAD_MAX)
    angular_unit = np.random.uniform(FIXED_ARC_ANG_MIN, FIXED_ARC_ANG_MAX)
    normalization_mode = np.random.choice([0, 1, 2])
    center_x = np.random.uniform(-1.0, 1.0)
    center_y = np.random.uniform(-1.0, 1.0)
    cfg.radial_unit = float(radial_unit)
    cfg.arc_size = float(angular_unit)
    cfg.center_x = float(center_x)
    cfg.center_y = float(center_y)
    cfg.normalization_mode = int(normalization_mode)
    split_size = radial_unit
    return cfg, split_size

@add_to_registry(split_variable_mapper, "RadialRepeatFixedArcBricked")
def var_resample_radial_repeat_fixed_arc_bricked(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_radial_unit():
        radial_unit = float(np.random.uniform(FIXED_ARC_RAD_MIN, FIXED_ARC_RAD_MAX))
        def edit_func(cfg):
            cfg.radial_unit = radial_unit
            return cfg
        return edit_func
    def replace_center():
        center_x = float(np.random.uniform(-1.0, 1.0))
        center_y = float(np.random.uniform(-1.0, 1.0))
        def edit_func(cfg):
            cfg.center_x = center_x
            cfg.center_y = center_y
            return cfg
        return edit_func
    def replace_angular_unit():
        angular_unit = float(np.random.uniform(FIXED_ARC_ANG_MIN, FIXED_ARC_ANG_MAX))
        def edit_func(cfg):
            cfg.arc_size = angular_unit
            return cfg
        return edit_func
    resamplers["radial_unit"] = replace_radial_unit
    resamplers["arc_size"] = replace_angular_unit
    resamplers["center_x"] = replace_center
    resamplers["center_y"] = replace_center
    resamplers["normalization_mode"] = None
    return resamplers

@add_to_registry(split_default_mapper, "RadialRepeatFixedArcBricked")
def default_radial_repeat_fixed_arc_bricked(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "RadialRepeatFixedArcBricked"
    cfg.radial_unit = 0.25
    cfg.arc_size = 0.35
    cfg.center_x = 0.0
    cfg.center_y = 0.0
    cfg.normalization_mode = 0
    return cfg

############ VoronoiRepeat ############

@add_to_registry(split_sampler_mapper, "VoronoiRepeat")
def sample_voronoi_repeat(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "VoronoiRepeat"
    x_size = np.random.uniform(X_SIZE_MIN, X_SIZE_MAX)
    y_size = np.random.uniform(X_SIZE_MIN, X_SIZE_MAX)
    noise_rate = np.random.uniform(VR_N_MIN, VR_N_MAX)
    # Norm - regular, ss_renorm, ss_renorm+rotate, secondary_renorm
    normalization_mode = 1# np.random.choice([1,])
    seed = np.random.randint(0, 100000)
    cfg.x_size = float(x_size)
    cfg.y_size = float(y_size)
    cfg.noise_rate = float(noise_rate)
    cfg.normalization_mode = int(normalization_mode)
    cfg.seed = int(seed)
    split_size = min(x_size, y_size)
    return cfg, split_size

@add_to_registry(split_variable_mapper, "VoronoiRepeat")
def var_resample_voronoi_repeat(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_x_size():
        x_size = float(np.random.uniform(X_SIZE_MIN, X_SIZE_MAX))
        def edit_func(cfg):
            cfg.x_size = x_size
            return cfg
        return edit_func
    def replace_y_size():
        y_size = float(np.random.uniform(X_SIZE_MIN, X_SIZE_MAX))
        def edit_func(cfg):
            cfg.y_size = y_size
            return cfg
        return edit_func
    def replace_noise_rate():
        noise_rate = float(np.random.uniform(VR_N_MIN, VR_N_MAX))
        def edit_func(cfg):
            cfg.noise_rate = noise_rate
            return cfg
        return edit_func
    def replace_normalization_mode():
        normalization_mode = int(np.random.choice([1, 3]))
        def edit_func(cfg):
            cfg.normalization_mode = normalization_mode
            return cfg
        return edit_func
    resamplers["x_size"] = replace_x_size
    resamplers["y_size"] = replace_y_size
    resamplers["noise_rate"] = replace_noise_rate
    resamplers["normalization_mode"] = None# replace_normalization_mode # if B is diff?
    resamplers["seed"] = None
    return resamplers

@add_to_registry(split_default_mapper, "VoronoiRepeat")
def default_voronoi_repeat(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "VoronoiRepeat"
    cfg.x_size = 0.25
    cfg.y_size = 0.25
    cfg.noise_rate = 0.15
    cfg.normalization_mode = 1
    cfg.seed = 0
    return cfg

############ IrregularRepeat ############

@add_to_registry(split_sampler_mapper, "IrregularRepeat")
def sample_irregular_repeat(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "IrregularRepeat"
    x_size = np.random.uniform(X_SIZE_MIN, X_SIZE_MAX)
    y_size = np.random.uniform(X_SIZE_MIN, X_SIZE_MAX)
    noise_rate = np.random.uniform(IN_N_MIN, IN_N_MAX)
    #  norm -> Is always screen space normed.
    seed = np.random.randint(0, 100000)
    cfg.x_size = float(x_size)
    cfg.y_size = float(y_size)
    cfg.noise_rate = float(noise_rate)
    cfg.seed = int(seed)
    cfg.normalization_mode = 1
    split_size = min(x_size, y_size)
    return cfg, split_size

@add_to_registry(split_variable_mapper, "IrregularRepeat")
def var_resample_irregular_repeat(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_x_size():
        x_size = float(np.random.uniform(X_SIZE_MIN, X_SIZE_MAX))
        def edit_func(cfg):
            cfg.x_size = x_size
            return cfg
        return edit_func
    def replace_y_size():
        y_size = float(np.random.uniform(X_SIZE_MIN, X_SIZE_MAX))
        def edit_func(cfg):
            cfg.y_size = y_size
            return cfg
        return edit_func
    def replace_noise_rate():
        noise_rate = float(np.random.uniform(IN_N_MIN, IN_N_MAX))
        def edit_func(cfg):
            cfg.noise_rate = noise_rate
            return cfg
        return edit_func
    resamplers["x_size"] = replace_x_size
    resamplers["y_size"] = replace_y_size
    resamplers["noise_rate"] = replace_noise_rate
    resamplers["seed"] = None
    resamplers["normalization_mode"] = None
    return resamplers

@add_to_registry(split_default_mapper, "IrregularRepeat")
def default_irregular_repeat(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "IrregularRepeat"
    cfg.x_size = 0.25
    cfg.y_size = 0.25
    cfg.noise_rate = 0.15
    cfg.seed = 0
    cfg.normalization_mode = 1
    return cfg

############ RandomFillRectRepeat ############

@add_to_registry(split_sampler_mapper, "RandomFillRectRepeat")
def sample_random_fill_rect_repeat(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "RandomFillRectRepeat"
    grid_size = np.random.uniform(X_SIZE_MIN, X_SIZE_MAX)
    seed = np.random.randint(0, 100000)
    cfg.seed = int(seed)
    split_size = grid_size
    return cfg, split_size

@add_to_registry(split_variable_mapper, "RandomFillRectRepeat")
def var_resample_random_fill_rect_repeat(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_grid_size():
        grid_size = float(np.random.uniform(X_SIZE_MIN, X_SIZE_MAX))
        def edit_func(cfg):
            cfg.grid_size = grid_size
            return cfg
        return edit_func
    resamplers["grid_size"] = replace_grid_size
    resamplers["fill_rate"] = None
    resamplers["seed"] = None
    return resamplers

@add_to_registry(split_default_mapper, "RandomFillRectRepeat")
def default_random_fill_rect_repeat(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "RandomFillRectRepeat"
    cfg.grid_size = 0.25
    cfg.fill_rate = 0.1
    cfg.seed = 0
    split_size = cfg.grid_size
    return cfg

############ RandomFillVoronoi ############
@add_to_registry(split_sampler_mapper, "RandomFillVoronoi")
def sample_random_fill_voronoi(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "RandomFillVoronoi"
    grid_size = np.random.uniform(X_SIZE_MIN, X_SIZE_MAX)
    seed = np.random.randint(0, 100000)
    cfg.seed = int(seed)
    split_size = grid_size
    return cfg, split_size

@add_to_registry(split_variable_mapper, "RandomFillVoronoi")
def var_resample_random_fill_voronoi(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_grid_size():
        grid_size = float(np.random.uniform(X_SIZE_MIN, X_SIZE_MAX))
        def edit_func(cfg):
            cfg.grid_size = grid_size
            return cfg
        return edit_func
    resamplers["grid_size"] = replace_grid_size
    resamplers["fill_rate"] = None
    resamplers["seed"] = None
    
    return resamplers

@add_to_registry(split_default_mapper, "RandomFillVoronoi")
def default_random_fill_voronoi(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "RandomFillVoronoi"
    cfg.grid_size = 0.25
    cfg.fill_rate = 0.1
    cfg.seed = 0
    return cfg


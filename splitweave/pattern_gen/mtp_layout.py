# LAYOUT -> (PRE_DEFORM) (DEFORM) (SPLIT) (POST_DEFORM)
from yacs.config import CfgNode as CN
import numpy as np
import torch as th
from .splitting_func import (split_default_mapper, split_sampler_mapper, split_variable_mapper)
from .deformation_func import (deform_default_mapper, deform_sampler_mapper, deform_variable_mapper)
from .utils import add_to_registry

layout_sampler_mapper = {}
layout_variable_mapper = {}
layout_default_mapper = {}

########### PRE_DEFORM ############

@add_to_registry(layout_sampler_mapper, "pre_deform")
def sample_pre_deform(parent_cfg, *args, **kwargs):
    cfg = CN()
    translation_x = np.random.uniform(-0.1, 0.1)
    translation_y = np.random.uniform(-0.1, -0.1)
    cfg.t_x = float(translation_x)
    cfg.t_y = float(translation_y)
    rotation = np.random.uniform(-np.pi/4, np.pi/4)

    cfg.rot = float(rotation)
    scale = np.random.uniform(0.8, 1.2) # Scale as a factor
    cfg.scale = float(scale)

    return cfg

@add_to_registry(layout_variable_mapper, "pre_deform")
def var_resampler_pre_deform(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_xy_size():
        translation_x = np.random.uniform(-0.1, 0.1)
        translation_y = np.random.uniform(-0.1, -0.1)
        def edit_func(cfg):
            cfg.t_x = translation_x
            cfg.t_y = translation_y
            return cfg
        return edit_func
    def replace_rot():
        rot = float(np.random.uniform(-np.pi/4, np.pi/4))
        def edit_func(cfg):
            cfg.rot = rot
            return cfg
        return edit_func
    def replace_scale():
        scale = float(np.random.uniform(0.8, 1.2))
        def edit_func(cfg):
            cfg.scale = scale
            return cfg
        return edit_func
    resamplers["t_x"] = None # replace_xy_size
    resamplers["t_y"] = None # replace_xy_size
    resamplers["rot"] = replace_rot
    resamplers["scale"] = None # replace_scale
    return resamplers

@add_to_registry(layout_default_mapper, "pre_deform")
def get_default_pre_deform(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg.t_x = 0.0
    cfg.t_y = 0.0
    cfg.rot = 0.0
    cfg.scale = 1.0
    return cfg

##### POST_DEFORM #####
@add_to_registry(layout_sampler_mapper, "post_deform")
def sample_post_deform(parent_cfg, split_size,  *args, **kwargs):
    cfg = CN()
    old_rot = parent_cfg.pre_deform.rot
    if np.random.choice([True, False]):
        rotation = -old_rot
    else:
        if np.abs(old_rot) > np.pi/4 and np.abs(old_rot) < 3*np.pi/4:
            rotation = -old_rot
        else:
            rotation = np.random.uniform(-np.pi/4, np.pi/4)
    cfg.rot = float(rotation)
    min_scale = 0.95 - 0.1 * split_size
    rescale = np.random.uniform(min_scale, 0.99)
    # If not Overlap. 
    cfg.scale = float(rescale)
    return cfg

@add_to_registry(layout_variable_mapper, "post_deform")
def var_resampler_post_deform(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_rot():
        old_rot = current_cfg.pre_deform.rot
        if np.random.choice([True, False]):
            rotation = -old_rot
        else:
            if np.abs(old_rot) > np.pi/4 and np.abs(old_rot) < 3*np.pi/4:
                rotation = -old_rot
            else:
                rotation = np.random.uniform(-np.pi/4, np.pi/4)
        def edit_func(cfg):
            cfg.rot = rotation
            return cfg
        return edit_func
    resamplers["rot"] = None
    resamplers["scale"] = None
    return resamplers

@add_to_registry(layout_default_mapper, "post_deform")
def get_default_post_deform(parent_cfg, *args, **kwargs):
    cfg = CN()
    old_rot = parent_cfg.pre_deform.rot
    rotation = -old_rot
    cfg.rot = rotation
    cfg.scale = 0.75
    return cfg

##### layout Function #####
SINGLE_TILE_SPLITS = [
    "RectRepeatFitting", "RandomFillRectRepeat", "RandomFillVoronoi",
]
NO_BG_SPLITS = [
     "RandomFillRectRepeat", "RandomFillVoronoi",
]

# Translate cell fx / tile fx between these. 
# BGX / Border / Fill can be arbitrary.
regular_layouts = [
    "RectRepeat",
    "RectRepeatFitting",
    "RectRepeatShiftedX",
    "RectRepeatShiftedY",
]
hexagonal_layouts = [
    "HexRepeat",
    "HexRepeatY"
]
# deform can be anything
# x, y, origin -> can be anything too. 
# 
radial_layouts = [
    "RadialRepeat",
    "RadialRepeatBricked",
    "RadialRepeatFixedArc",
    "RadialRepeatFixedArcBricked"
]
irregular_layouts = [
    "VoronoiRepeat",
    "IrregularRepeat",
]
random_fill = [
    "RandomFillRectRepeat",
    "RandomFillVoronoi"
]

overlap_layouts = [
    "OverlapX", "OverlapY", "OverlapXY"
]
META_SPLIT_MAPPER = {}
META_SPLIT_MAPPER.update({x: "regular" for x in regular_layouts})
META_SPLIT_MAPPER.update({x: "hexagonal" for x in hexagonal_layouts})
META_SPLIT_MAPPER.update({x: "radial" for x in radial_layouts})
META_SPLIT_MAPPER.update({x: "irregular" for x in irregular_layouts})
META_SPLIT_MAPPER.update({x: "random_fill" for x in random_fill})
META_SPLIT_MAPPER.update({x: "overlap" for x in overlap_layouts})

# @add_to_registry(layout_sampler_mapper, "layout")
def sample_layout(n_tiles, *args, **kwargs):
    cfg = CN()
    # 50% do default
    if np.random.choice([True, False]):
        cfg.pre_deform = get_default_pre_deform(cfg, *args, **kwargs)
    else:
        cfg.pre_deform = sample_pre_deform(cfg, *args, **kwargs)
    if np.random.choice([True, False]):
        cfg.deform = deform_default_mapper['no_deform'](cfg, *args, **kwargs)
    else:
        deform_types = list(deform_sampler_mapper.keys())
        cfg.deform = deform_sampler_mapper[np.random.choice(deform_types)](cfg, *args, **kwargs)
    # Always have a random splitting function.
    split_types = list(split_sampler_mapper.keys())

    if n_tiles > 1:
        split_types = [x for x in split_types if not x in SINGLE_TILE_SPLITS]
    if n_tiles > 2:
        remove_types = ["OverlapX", "OverlapY", "OverlapXY"]
        split_types = [x for x in split_types if not x in remove_types]
    # Simple_types:
    # split_types = ["RadialRepeatFixedArcBricked"]
    cfg.split, split_size = split_sampler_mapper[np.random.choice(split_types)](cfg, *args, **kwargs)
    
    if np.random.choice([True, False]):
        cfg.post_deform = get_default_post_deform(cfg, *args, **kwargs)
    else:
        cfg.post_deform = sample_post_deform(cfg, split_size=split_size, *args, **kwargs)
    return cfg

# @add_to_registry(layout_variable_mapper, "layout")
def var_resampler_layout(current_cfg, *args, **kwargs):
    resamplers = {}
    # Disable predeform, 
    # this should be only if the layout 
    def replace_pre_deform():
        new_pre_deform = sample_pre_deform(current_cfg)
        def edit_func(cfg):
            cfg.pre_deform = new_pre_deform
            return cfg
        return edit_func
    def replace_deform():
        deform_types = list(deform_sampler_mapper.keys())
        deform = deform_sampler_mapper[np.random.choice(deform_types)](current_cfg)
        def edit_func(cfg):
            cfg.deform = deform
            return cfg
        return edit_func
    def replace_split():
        split_types = list(split_sampler_mapper.keys())
        # get the current tile_ratio
        if hasattr(current_cfg.split, "tile_ratio"):
            tile_ratio = current_cfg.split.tile_ratio
        else:
            tile_ratio = (1, 1)
        split = split_sampler_mapper[np.random.choice(split_types)](current_cfg, tile_ratio=tile_ratio)
        def edit_func(cfg):
            cfg.split = split
            return cfg
        return edit_func
    def replace_post_deform():
        post_deform = sample_post_deform(current_cfg)
        def edit_func(cfg):
            cfg.post_deform = post_deform
            return cfg
        return edit_func
    resamplers["pre_deform"] = None # replace_pre_deform
    resamplers["deform"] = replace_deform
    resamplers["split"] = replace_split
    resamplers["post_deform"] = None # replace_post_deform# replace_post_deform
    # For the params inside each.
    pre_deform_samplers = var_resampler_pre_deform(current_cfg)
    for key, value in pre_deform_samplers.items():
        new_key = f"pre_deform.{key}"
        if value is None:
            resamplers[new_key] = None
        else:
            def wrapper_outer(key=key):
                inner_func = pre_deform_samplers[key]()
                def wrapper_func(cfg):
                    cfg.pre_deform = inner_func(cfg.pre_deform)
                    return cfg
                return wrapper_func
            resamplers[new_key] = wrapper_outer
    
    split_type = current_cfg.split._type
    split_resamplers = split_variable_mapper[split_type](current_cfg.split)
    for key, value in split_resamplers.items():
        new_key = f"split.{key}"
        if value is None:
            resamplers[new_key] = None
        else:
            def wrapper_outer(key=key):
                inner_func = split_resamplers[key]()
                def wrapper_func(cfg):
                    cfg.split = inner_func(cfg.split)
                    return cfg
                return wrapper_func
            resamplers[new_key] = wrapper_outer
    return resamplers

# @add_to_registry(layout_default_mapper, "layout")
def get_default_layout(*args, **kwargs):
    cfg = CN()
    cfg.pre_deform = get_default_pre_deform(cfg, *args, **kwargs)
    cfg.deform = deform_default_mapper['no_deform'](cfg, *args, **kwargs)
    cfg.split = split_default_mapper["RectRepeat"](cfg, *args, **kwargs)
    cfg.post_deform = get_default_post_deform(cfg, *args, **kwargs)
    return cfg


def bg_sample_pre_deform(parent_cfg, *args, **kwargs):
    cfg = CN()
    translation_x = np.random.uniform(-2, 2)
    translation_y = np.random.uniform(-2, 2)
    cfg.t_x = float(translation_x)
    cfg.t_y = float(translation_y)
    rotation = np.random.uniform(-np.pi, np.pi)

    cfg.rot = float(rotation)
    scale = np.random.uniform(0.2, 0.75) # Scale as a factor
    cfg.scale = float(scale)

    return cfg


def shift_sample_pre_deform(orignal_cfg, *args, **kwargs):
    cfg = CN()
    translation_x = np.random.uniform(-0.1, 0.1) + orignal_cfg.t_x
    translation_y = np.random.uniform(-0.1, 0.1) + orignal_cfg.t_y
    cfg.t_x = float(translation_x)
    cfg.t_y = float(translation_y)
    rotation = np.random.uniform(-np.pi/8, np.pi/8) + orignal_cfg.rot

    cfg.rot = float(rotation)
    scale = np.random.uniform(-0.1, 0.1) + orignal_cfg.scale # Scale as a factor
    cfg.scale = float(scale)

    return cfg
# @add_to_registry(layout_sampler_mapper, "layout")
def bg_sample_layout(n_tiles, *args, **kwargs):
    cfg = CN()
    # 50% do default
    cfg.pre_deform = bg_sample_pre_deform(cfg, *args, **kwargs)
    if np.random.choice([True, False], p=[2/3, 1/3]):
        cfg.deform = deform_default_mapper['no_deform'](cfg, *args, **kwargs)
    else:
        deform_types = list(deform_sampler_mapper.keys())
        cfg.deform = deform_sampler_mapper[np.random.choice(deform_types)](cfg, *args, **kwargs)
    # Always have a random splitting function.
    split_types = ["RectRepeat", "HexRepeat", "HexRepeatY", 
                    "RectRepeatShiftedX", "RectRepeatShiftedY",
                    "VoronoiRepeat",] # "IrregularRepeat"
    # split_types = ["OverlapXY"]
    cfg.split, split_size = split_sampler_mapper[np.random.choice(split_types)](cfg, *args, **kwargs)

    if np.random.choice([True, False]):
        cfg.post_deform = get_default_post_deform(cfg, *args, **kwargs)
    else:
        cfg.post_deform = sample_post_deform(cfg, split_size=split_size, *args, **kwargs)
    return cfg


# @add_to_registry(layout_sampler_mapper, "layout")
def fill_sample_layout(n_tiles, *args, **kwargs):
    cfg = CN()
    # 50% do default
    cfg.pre_deform = bg_sample_pre_deform(cfg, *args, **kwargs)
    cfg.deform = deform_default_mapper['no_deform'](cfg, *args, **kwargs)
    # Always have a random splitting function.
    split_types = ["RectRepeat", "HexRepeat", "HexRepeatY", 
                   "RectRepeatShiftedX", "RectRepeatShiftedY",
                #    "VoronoiRepeat", "IrregularRepeat"
                   ] # "IrregularRepeat"
    # split_types = ["OverlapXY"]
    cfg.split, split_size = split_sampler_mapper[np.random.choice(split_types)](cfg, *args, **kwargs)

    if np.random.choice([True, False]):
        cfg.post_deform = get_default_post_deform(cfg, *args, **kwargs)
    else:
        cfg.post_deform = sample_post_deform(cfg, split_size=split_size, *args, **kwargs)
    return cfg

# Or do the random filling -> Do two versions. 


# @add_to_registry(layout_sampler_mapper, "layout")
def random_fill_sample_layout(split_types, *args, **kwargs):
    cfg = CN()
    # 50% do default
    if np.random.choice([True, False]):
        cfg.pre_deform = get_default_pre_deform(cfg, *args, **kwargs)
    else:
        cfg.pre_deform = sample_pre_deform(cfg, *args, **kwargs)
    cfg.deform = deform_default_mapper['no_deform'](cfg, *args, **kwargs)
    # NO Deform
    cfg.split, split_size = split_sampler_mapper[np.random.choice(split_types)](cfg, *args, **kwargs)

    if np.random.choice([True, False]):
        cfg.post_deform = get_default_post_deform(cfg, *args, **kwargs)
    else:
        cfg.post_deform = sample_post_deform(cfg, split_size=split_size, *args, **kwargs)
    return cfg

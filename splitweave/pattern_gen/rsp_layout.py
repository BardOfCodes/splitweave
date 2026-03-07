# LAYOUT -> (PRE_DEFORM) (DEFORM) (SPLIT) (POST_DEFORM)
from yacs.config import CfgNode as CN
import numpy as np
import torch as th
from .rsp_splits import (split_default_mapper, split_sampler_mapper)
from .deformation_func import (deform_default_mapper, deform_sampler_mapper)
from .utils import add_to_registry
from .mtp_tile import sample_retro_tile

layout_sampler_mapper = {}
layout_variable_mapper = {}
layout_default_mapper = {}

########### PRE_DEFORM ############

@add_to_registry(layout_sampler_mapper, "pre_deform")
def sample_pre_deform(parent_cfg, *args, **kwargs):
    cfg = CN()
    translation_x = np.random.uniform(-1, 1)
    translation_y = np.random.uniform(-1, 1)
    cfg.t_x = float(translation_x)
    cfg.t_y = float(translation_y)
    rotation = np.random.uniform(-np.pi, np.pi)
    cfg.rot = float(rotation)
    scale = np.random.uniform(0.5, 1.5) # Scale as a factor
    cfg.scale = float(scale)
    return cfg

@add_to_registry(layout_default_mapper, "pre_deform")
def get_default_pre_deform(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg.t_x = 0.0
    cfg.t_y = 0.0
    cfg.rot = 0.0
    cfg.scale = 1.0
    return cfg

# @add_to_registry(layout_sampler_mapper, "layout")
def sample_layout(*args, **kwargs):
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
    if np.random.choice([True, False]):
        origin_norm = np.linalg.norm([cfg.pre_deform.t_x, cfg.pre_deform.t_y])
        if origin_norm > 0.7:
            cfg.foreground = None
            has_foreground = False
        else:
            foreground = split_sampler_mapper['shape_sdf'](cfg, *args, **kwargs)
            cfg.foreground = foreground
            # counts
            cfg.foreground.growth_mode = "linear"
            ring_counts = np.random.randint(1, 4)
            cfg.foreground.ring_counts = int(ring_counts)
            has_foreground = True
    else:
        cfg.foreground = None
        has_foreground = False
    split_types = list(split_sampler_mapper.keys())
    if has_foreground:
        split_types.remove("shape_overlay")
        split_types.remove("shape_sdf")

    # Simple_types:
    cfg.split = split_sampler_mapper[np.random.choice(split_types)](cfg, *args, **kwargs)
    
    return cfg

# @add_to_registry(layout_default_mapper, "layout")
def get_default_layout(*args, **kwargs):
    cfg = CN()
    cfg.pre_deform = get_default_pre_deform(cfg, *args, **kwargs)
    cfg.deform = deform_default_mapper['no_deform'](cfg, *args, **kwargs)
    cfg.split = split_default_mapper["RectRepeat"](cfg, *args, **kwargs)
    return cfg

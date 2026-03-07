import numpy as np
from yacs.config import CfgNode as CN
from ..pattern_gen.rsp_layout import sample_layout
from ..pattern_gen.rsp_color import sample_fill_cfg
from ..pattern_gen.rsp_main import sample_rsp_pattern
from ..pattern_gen.rsp_splits import split_sampler_mapper
NO_FOREGROUND_TYPES = ["shape_overlay", "shape_sdf"]

def rsp_replace_layout(*args, **kwargs):
    pattern_a = sample_rsp_pattern(*args, **kwargs)
    pattern_b = sample_rsp_pattern(*args, **kwargs)
    # keep the foreground?
    new_layout = sample_layout(*args, **kwargs)
    pattern_a_star = pattern_a.clone()
    pattern_a_star.layout = new_layout.clone()
    pattern_b_star = pattern_b.clone()
    pattern_b_star.layout = new_layout.clone()

    output_dict = {
        "patterns_a": [pattern_a],
        "patterns_b": [pattern_b],
        "patterns_a_star": [pattern_a_star],
        "patterns_b_star": [pattern_b_star],
    }
    return output_dict

def rsp_replace_coloring(*args, **kwargs):
    pattern_a = sample_rsp_pattern(*args, **kwargs)
    pattern_b = sample_rsp_pattern(*args, **kwargs)

    new_fill = sample_fill_cfg()
    pattern_a_star = pattern_a.clone()
    pattern_a_star.fill = new_fill.clone()
    pattern_b_star = pattern_b.clone()
    pattern_b_star.fill = new_fill.clone()

    output_dict = {
        "patterns_a": [pattern_a],
        "patterns_b": [pattern_b],
        "patterns_a_star": [pattern_a_star],
        "patterns_b_star": [pattern_b_star],
    }
    return output_dict

def replace_foreground(*args, **kwargs):

    pattern_found = False
    while not pattern_found:
        pattern_a = sample_rsp_pattern(*args, **kwargs)
        pattern_b = sample_rsp_pattern(*args, **kwargs)
        a_type = pattern_a.layout.split._type
        b_type = pattern_b.layout.split._type
        if a_type in NO_FOREGROUND_TYPES or b_type in NO_FOREGROUND_TYPES:
            pattern_found = False
        else:
            cfg = pattern_a.layout
            origin_norm_a = np.linalg.norm([cfg.pre_deform.t_x, cfg.pre_deform.t_y])
            cfg = pattern_b.layout
            origin_norm_b = np.linalg.norm([cfg.pre_deform.t_x, cfg.pre_deform.t_y])

            if origin_norm_a > 0.7 or origin_norm_b > 0.7:
                pattern_found = False
            else:
                pattern_found = True
    # Could be removal as well -> For removal they should have it originally. 
    
    new_foreground = split_sampler_mapper['shape_sdf'](None, *args, **kwargs)
    ring_counts = np.random.randint(1, 4)
    new_foreground.ring_counts = int(ring_counts)

    
    pattern_a_star = pattern_a.clone()
    pattern_a_star.layout.foreground = new_foreground.clone()
    pattern_b_star = pattern_b.clone()
    pattern_b_star.layout.foreground = new_foreground.clone()

    output_dict = {
        "patterns_a": [pattern_a],
        "patterns_b": [pattern_b],
        "patterns_a_star": [pattern_a_star],
        "patterns_b_star": [pattern_b_star],
    }
    return output_dict



def replace_background(*args, **kwargs):

    pattern_found = False
    while not pattern_found:
        pattern_a = sample_rsp_pattern(*args, **kwargs)
        pattern_b = sample_rsp_pattern(*args, **kwargs)
        a_foreground = pattern_a.layout.foreground
        b_foreground = pattern_b.layout.foreground
        if a_foreground is not None and b_foreground is not None:
            new_layout = sample_layout(*args, **kwargs)
            new_layout_type = new_layout.split._type
            if not new_layout_type in NO_FOREGROUND_TYPES:
                pattern_found = True
            else:
                pattern_found = False
        else:
            pattern_found = False
        
    
    
    pattern_a_star = pattern_a.clone()
    pattern_a_star.layout = new_layout.clone()
    pattern_a_star.layout.foreground = a_foreground.clone()
    pattern_a_star.layout.pre_deform = pattern_a.layout.pre_deform.clone()
    pattern_b_star = pattern_b.clone()
    pattern_b_star.layout = new_layout.clone()
    pattern_b_star.layout.foreground = b_foreground.clone()
    pattern_b_star.layout.pre_deform = pattern_b.layout.pre_deform.clone()

    output_dict = {
        "patterns_a": [pattern_a],
        "patterns_b": [pattern_b],
        "patterns_a_star": [pattern_a_star],
        "patterns_b_star": [pattern_b_star],
    }
    return output_dict

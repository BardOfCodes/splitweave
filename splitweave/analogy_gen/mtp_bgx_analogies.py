import numpy as np

from ..pattern_gen.mtp_tile import sample_tileset
from ..pattern_gen.mtp_main import sample_mtp_pattern, sample_default_mtp_pattern, NON_BG, NON_BORDERABLE, NON_FILLABLE, NO_EFFECT_TYPE, PARTIAL_EFFECT_TYPE
from ..pattern_gen.mtp_cellfx import sample_tile_ordering_fx
from ..pattern_gen.mtp_bgx import sample_bg, sample_border, default_border, sample_default_bg
from ..pattern_gen.mtp_layout import META_SPLIT_MAPPER
from ..pattern_gen.splitting_func import split_default_mapper
from ..pattern_gen.mtp_cellfx import sample_mtp_cellfx
from ..pattern_gen.filling_function import random_fill_rect_repeat, sample_fill, sample_repeat_fill
# Completely replace the layout


def bg_change(filenames, filename_to_indices, harder_constraint=0, old_flag=False, *args, **kwargs):
    # Lets create 2 sets.

    # Set tile order aside
    found_pattern = False
    resample_both = True
    while not found_pattern:
        if resample_both:
            pattern_a = sample_mtp_pattern(filenames, filename_to_indices)
            pattern_b = sample_mtp_pattern(filenames, filename_to_indices)
        else:
            pattern_b = sample_mtp_pattern(filenames, filename_to_indices)

        a_type = pattern_a.layout_cfg.split._type
        b_type = pattern_b.layout_cfg.split._type
        fill_a = pattern_a.fill_cfg
        fill_b = pattern_b.fill_cfg
        type_invalid = a_type in NON_BG or b_type in NON_BG
        fill_invalid = (fill_a is not None) or (fill_b is not None)
        if type_invalid or fill_invalid:
            # also avoid bg with patterns
            found_pattern = False
            resample_both = True
        else:
            if harder_constraint == 1:
                meta_split_a = META_SPLIT_MAPPER[a_type]
                meta_split_b = META_SPLIT_MAPPER[b_type]
                if meta_split_a == meta_split_b:
                    found_pattern = True
                else:
                    found_pattern = False
                    resample_both = False # keep probability same
            elif harder_constraint == 2:
                meta_split_a = a_type
                meta_split_b = b_type
                if meta_split_a == meta_split_b:
                    found_pattern = True
                else:
                    found_pattern = False
                    resample_both = False # keep probability same
            elif harder_constraint in [3, 4]:
                # set a and b to have the same layout
                pattern_b.layout_cfg = pattern_a.layout_cfg.clone()
                found_pattern = True
            else:
                found_pattern = True

    pattern_c = sample_default_mtp_pattern(filenames, filename_to_indices, n_tiles=1)

    new_bg_cfg = sample_bg()

    pattern_a_star = pattern_a.clone()
    pattern_a_star.bg_cfg = new_bg_cfg.clone()

    pattern_b_star = pattern_b.clone()
    pattern_b_star.bg_cfg = new_bg_cfg.clone()

    pattern_c_star = pattern_c.clone()
    pattern_c_star.bg_cfg = new_bg_cfg.clone()

    output_dict = {
        "patterns_a": [pattern_a],
        "patterns_b": [pattern_b],
        "patterns_c": [pattern_c],
        "patterns_a_star": [pattern_a_star],
        "patterns_b_star": [pattern_b_star],
        "patterns_c_star": [pattern_c_star],
    }
    return output_dict

def border_change(filenames, filename_to_indices, old_flag=False, *args, **kwargs):
    # Lets create 2 sets.


    # Set tile order aside
    # Set tile order aside
    found_pattern = False
    while not found_pattern:
        pattern_a = sample_mtp_pattern(filenames, filename_to_indices)
        pattern_b = sample_mtp_pattern(filenames, filename_to_indices)
        a_type = pattern_a.layout_cfg.split._type
        b_type = pattern_b.layout_cfg.split._type
        if a_type in NON_BORDERABLE or b_type in NON_BORDERABLE:
            # also avoid bg with patterns
            found_pattern = False
        else:
            a_bg_type = pattern_a.bg_cfg.bg_mode
            b_bg_type = pattern_b.bg_cfg.bg_mode
            if a_bg_type == "plain" and b_bg_type == "plain":
                found_pattern = True
            else:
                found_pattern = False
    
    
    pattern_c = sample_default_mtp_pattern(filenames, filename_to_indices, n_tiles=1)
    
    if np.random.choice([True, False], p=[0.25, 0.75]):
        pattern_a_star = pattern_a.clone()
        pattern_a_star.border_cfg = None

        pattern_b_star = pattern_b.clone()
        pattern_b_star.border_cfg = None
        pattern_c_star = pattern_c.clone()
        pattern_c_star.border_cfg = None
    else:
        new_bg_cfg = sample_border(pattern_a.layout_cfg)
        pattern_c.border_cfg = None
        pattern_a_star = pattern_a.clone()
        pattern_a_star.border_cfg = new_bg_cfg.clone()

        pattern_b_star = pattern_b.clone()
        pattern_b_star.border_cfg = new_bg_cfg.clone()
        if 'exec_seed' in pattern_b.layout_cfg.split.keys():
            pattern_b_star.border_cfg.exec_seed = pattern_b.layout_cfg.split.seed
        
        pattern_c_star = pattern_c.clone()
        pattern_c_star.border_cfg = new_bg_cfg.clone()
    #  Will not need to set the seed here.
    

    output_dict = {
        "patterns_a": [pattern_a],
        "patterns_b": [pattern_b],
        "patterns_c": [pattern_c],
        "patterns_a_star": [pattern_a_star],
        "patterns_b_star": [pattern_b_star],
        "patterns_c_star": [pattern_c_star],
    }
    return output_dict
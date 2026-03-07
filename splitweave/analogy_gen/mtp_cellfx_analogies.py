import numpy as np

from ..pattern_gen.mtp_tile import sample_tileset
from ..pattern_gen.mtp_main import sample_mtp_pattern, sample_default_mtp_pattern, NON_BORDERABLE, NON_FILLABLE, NO_EFFECT_TYPE, PARTIAL_EFFECT_TYPE
from ..pattern_gen.mtp_cellfx import sample_tile_ordering_fx
from ..pattern_gen.mtp_bgx import sample_bg, sample_border, default_border, sample_default_bg
from ..pattern_gen.mtp_layout import get_default_layout, sample_layout, sample_pre_deform, SINGLE_TILE_SPLITS, split_sampler_mapper, deform_sampler_mapper
from ..pattern_gen.splitting_func import split_default_mapper
from ..pattern_gen.mtp_layout import META_SPLIT_MAPPER
from ..pattern_gen.mtp_cellfx import sample_mtp_cellfx, cellfx_sampler_mapper
from ..pattern_gen.filling_function import random_fill_rect_repeat, sample_fill, sample_repeat_fill
# Completely replace the layout

def full_cellfx_change(filenames, filename_to_indices, harder_constraint=0, old_flag=False, *args, **kwargs):
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
        if a_type in NO_EFFECT_TYPE or b_type in NO_EFFECT_TYPE:
            found_pattern = False
            resample_both = True
        else:
            if harder_constraint==1: 
                # the higher class of a and b split hsould match
                meta_split_a = META_SPLIT_MAPPER[a_type]
                meta_split_b = META_SPLIT_MAPPER[b_type]
                if meta_split_a == meta_split_b:
                    found_pattern = True
                else:
                    found_pattern = False
                    resample_both = False # keep probability same
            elif harder_constraint==2:
                # set a and b to have the same layout
                # the higher class of a and b split hsould match
                meta_split_a = a_type
                meta_split_b = b_type
                if meta_split_a == meta_split_b:
                    found_pattern = True
                else:
                    found_pattern = False
                    resample_both = False # keep probability same

            elif harder_constraint == 3:
                # set a and b to have the same layout
                pattern_b.layout_cfg = pattern_a.layout_cfg.clone()
                found_pattern = True
            elif harder_constraint == 4:
                # should the tile order also match?
                pattern_b.layout_cfg = pattern_a.layout_cfg.clone()
                n_tiles_a = len(pattern_a.tile_cfg.tileset)
                n_tiles_b = len(pattern_b.tile_cfg.tileset)
                if n_tiles_a == n_tiles_b:
                    pattern_b.cellfx_cfg.tile_order = pattern_a.cellfx_cfg.tile_order.clone()
                found_pattern = True
            else:
                found_pattern = True

        
    
    pattern_c = sample_default_mtp_pattern(filenames, filename_to_indices, n_tiles=1)

    if a_type in PARTIAL_EFFECT_TYPE or b_type in PARTIAL_EFFECT_TYPE:
        partial_effects = True
    else:
        partial_effects = False
    new_cellfx = sample_mtp_cellfx(n_tiles=1, partial_effects=partial_effects, allow_nil=False)

    pattern_a_star = pattern_a.clone()
    pattern_a_star.cellfx_cfg = new_cellfx.clone()
    pattern_a_star.cellfx_cfg.tile_order = pattern_a.cellfx_cfg.tile_order.clone()

    pattern_b_star = pattern_b.clone()
    pattern_b_star.cellfx_cfg = new_cellfx.clone()
    pattern_b_star.cellfx_cfg.tile_order = pattern_b.cellfx_cfg.tile_order.clone()

    pattern_c_star = pattern_c.clone()
    pattern_c_star.cellfx_cfg = new_cellfx.clone()
    pattern_c_star.cellfx_cfg.tile_order = pattern_c.cellfx_cfg.tile_order.clone()

    output_dict = {
        "patterns_a": [pattern_a],
        "patterns_b": [pattern_b],
        "patterns_c": [pattern_c],
        "patterns_a_star": [pattern_a_star],
        "patterns_b_star": [pattern_b_star],
        "patterns_c_star": [pattern_c_star],
    }
    return output_dict

def add_cellfx(filenames, filename_to_indices, harder_constraint=0, old_flag=False, *args, **kwargs):
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
        if not a_type in NO_EFFECT_TYPE and not b_type in NO_EFFECT_TYPE:

            n_effects_a = len(pattern_a.cellfx_cfg.effects)
            n_effects_b = len(pattern_b.cellfx_cfg.effects)
            n_effects = max(n_effects_a, n_effects_b)
            if n_effects < 3:
                if a_type in PARTIAL_EFFECT_TYPE or b_type in PARTIAL_EFFECT_TYPE:
                    partial_effects = True
                else:
                    partial_effects = False
                new_cellfx = sample_mtp_cellfx(n_tiles=1, partial_effects=partial_effects, allow_nil=False,
                                               avoid_t=True)
                new_effect_type = new_cellfx.effects[0]._type
                in_a = new_effect_type in [effect._type for effect in pattern_a.cellfx_cfg.effects]
                in_b = new_effect_type in [effect._type for effect in pattern_b.cellfx_cfg.effects]
                if not in_a and not in_b:
                    if harder_constraint == 1: 
                        meta_split_a = META_SPLIT_MAPPER[a_type]
                        meta_split_b = META_SPLIT_MAPPER[b_type]
                        if meta_split_a == meta_split_b:
                            found_pattern = True
                        else:
                            found_pattern = False
                            resample_both = False # keep probability same
                    elif harder_constraint == 2:
                        # set a and b to have the same layout
                        meta_split_a = a_type
                        meta_split_b = b_type
                        if meta_split_a == meta_split_b:
                            found_pattern = True
                        else:
                            found_pattern = False
                            resample_both = False # keep probability same

                    elif harder_constraint == 3:
                        # set a and b to have the same layout
                        pattern_b.layout_cfg = pattern_a.layout_cfg.clone()
                        found_pattern = True
                    elif harder_constraint == 4:
                        # should the tile order also match?
                        pattern_b.layout_cfg = pattern_a.layout_cfg.clone()
                        if len(pattern_a.tile_cfg.tileset) == len(pattern_b.tile_cfg.tileset):
                            pattern_b.cellfx_cfg.tile_order = pattern_a.cellfx_cfg.tile_order.clone()
                        found_pattern = True
                    else:
                        found_pattern = True
        else:
            found_pattern = False
            resample_both = True
    
    pattern_c = sample_default_mtp_pattern(filenames, filename_to_indices, n_tiles=1)
    new_effect = new_cellfx.effects[0]
    pattern_a_star = pattern_a.clone()
    pattern_a_star.cellfx_cfg.effects.append(new_effect)
    pattern_b_star = pattern_b.clone()
    pattern_b_star.cellfx_cfg.effects.append(new_effect)
    pattern_c_star = pattern_c.clone()
    pattern_c_star.cellfx_cfg.effects.append(new_effect)
    output_dict = {
        "patterns_a": [pattern_a],
        "patterns_b": [pattern_b],
        "patterns_c": [pattern_c],
        "patterns_a_star": [pattern_a_star],
        "patterns_b_star": [pattern_b_star],
        "patterns_c_star": [pattern_c_star],
    }
    return output_dict

def remove_cellfx(filenames, filename_to_indices, harder_constraint=0, old_flag=False, *args, **kwargs):
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
        if not a_type in NO_EFFECT_TYPE and not b_type in NO_EFFECT_TYPE:
            n_effects_a = len(pattern_a.cellfx_cfg.effects)
            n_effects_b = len(pattern_b.cellfx_cfg.effects)
            n_effects = max(n_effects_a, n_effects_b)
            if n_effects < 3:
                if a_type in PARTIAL_EFFECT_TYPE or b_type in PARTIAL_EFFECT_TYPE:
                    partial_effects = True
                else:
                    partial_effects = False
                new_cellfx = sample_mtp_cellfx(n_tiles=1, partial_effects=partial_effects, allow_nil=False,
                                               avoid_t=True)
                new_effect_type = new_cellfx.effects[0]._type
                in_a = new_effect_type in [effect._type for effect in pattern_a.cellfx_cfg.effects]
                in_b = new_effect_type in [effect._type for effect in pattern_b.cellfx_cfg.effects]
                if not in_a and not in_b:
                    if harder_constraint == 1:
                        meta_split_a = META_SPLIT_MAPPER[a_type]
                        meta_split_b = META_SPLIT_MAPPER[b_type]
                        if meta_split_a == meta_split_b:
                            found_pattern = True
                        else:
                            found_pattern = False
                            resample_both = False # keep probability same
                    elif harder_constraint == 2:
                        # set a and b to have the same layout
                        meta_split_a = a_type
                        meta_split_b = b_type
                        if meta_split_a == meta_split_b:
                            found_pattern = True
                        else:
                            found_pattern = False
                            resample_both = False # keep probability same

                    elif harder_constraint == 3:
                        # set a and b to have the same layout
                        pattern_b.layout_cfg = pattern_a.layout_cfg.clone()
                        found_pattern = True
                        
                    elif harder_constraint == 4:
                        # should the tile order also match?
                        pattern_b.layout_cfg = pattern_a.layout_cfg.clone()
                        if len(pattern_a.tile_cfg.tileset) == len(pattern_b.tile_cfg.tileset):
                            pattern_b.cellfx_cfg.tile_order = pattern_a.cellfx_cfg.tile_order.clone()
                        found_pattern = True
                    else:
                        found_pattern = True
        else:
            found_pattern = False
            resample_both = True
    
    pattern_c = sample_default_mtp_pattern(filenames, filename_to_indices, n_tiles=1)
    new_effect = new_cellfx.effects[0]
    pattern_a_star = pattern_a.clone()
    pattern_a_star.cellfx_cfg.effects.append(new_effect)
    pattern_b_star = pattern_b.clone()
    pattern_b_star.cellfx_cfg.effects.append(new_effect)
    pattern_c_star = pattern_c.clone()
    pattern_c_star.cellfx_cfg.effects.append(new_effect)
    output_dict = {
        "patterns_a": [pattern_a_star],
        "patterns_b": [pattern_b_star],
        "patterns_c": [pattern_c_star],
        "patterns_a_star": [pattern_a],
        "patterns_b_star": [pattern_b],
        "patterns_c_star": [pattern_c],
    }
    return output_dict

# Def replace effect
# SKIP
def replace_cellfx(filenames, filename_to_indices, old_flag=False, *args, **kwargs):
    # Lets create 2 sets.

    # Set tile order aside
    found_pattern = False
    while not found_pattern:
        pattern_a = sample_mtp_pattern(filenames, filename_to_indices)
        pattern_b = sample_mtp_pattern(filenames, filename_to_indices)
        # Make one of the effects shared and edit. 

        a_type = pattern_a.layout_cfg.split._type
        b_type = pattern_b.layout_cfg.split._type
        if not a_type in NO_EFFECT_TYPE and not b_type in NO_EFFECT_TYPE:
            n_effects_a = len(pattern_a.cellfx_cfg.effects)
            n_effects_b = len(pattern_b.cellfx_cfg.effects)
            n_effects = min(n_effects_a, n_effects_b)
            if n_effects >= 1:
                rand_a = np.random.choice(n_effects_a)
                rand_b = np.random.choice(n_effects_b)
                pattern_a.cellfx_cfg.effects.pop(rand_a)
                pattern_b.cellfx_cfg.effects.pop(rand_b)
                if a_type in PARTIAL_EFFECT_TYPE or b_type in PARTIAL_EFFECT_TYPE:
                    partial_effects = True
                else:
                    partial_effects = False
                new_cellfx = sample_mtp_cellfx(n_tiles=1, partial_effects=partial_effects, allow_nil=False,
                                               avoid_t=True)
                new_effect_type = new_cellfx.effects[0]._type
                in_a = new_effect_type in [effect._type for effect in pattern_a.cellfx_cfg.effects]
                in_b = new_effect_type in [effect._type for effect in pattern_b.cellfx_cfg.effects]
                if not in_a and not in_b:
                    # now sample 3 of an effect, allot to ac , b, a*b*c*
                    effect_name = new_effect_type
                    effect_cfg_a = cellfx_sampler_mapper[effect_name](None)
                    effect_cfg_b = cellfx_sampler_mapper[effect_name](None)
                    effect_cfg_diff = cellfx_sampler_mapper[effect_name](None)
                    found_pattern = True
                else:
                    found_pattern = False
        else:
            found_pattern = False
    
    pattern_c = sample_default_mtp_pattern(filenames, filename_to_indices, n_tiles=1)
    
    pattern_a.cellfx_cfg.effects.append(effect_cfg_a)
    pattern_b.cellfx_cfg.effects.append(effect_cfg_b)
    pattern_c.cellfx_cfg.effects.append(effect_cfg_a)

    pattern_a_star = pattern_a.clone()
    pattern_a_star.cellfx_cfg.effects = [x.clone() for x in pattern_a.cellfx_cfg.effects]
    pattern_a_star.cellfx_cfg.effects[-1] = effect_cfg_diff.clone()
    pattern_b_star = pattern_b.clone()
    pattern_b_star.cellfx_cfg.effects = [x.clone() for x in pattern_b.cellfx_cfg.effects]
    pattern_b_star.cellfx_cfg.effects[-1] = effect_cfg_diff.clone()

    pattern_c_star = pattern_c.clone()
    pattern_c_star.cellfx_cfg.effects = [x.clone() for x in pattern_c.cellfx_cfg.effects]
    pattern_c_star.cellfx_cfg.effects[-1] = effect_cfg_diff.clone()

    output_dict = {
        "patterns_a": [pattern_a_star],
        "patterns_b": [pattern_b_star],
        "patterns_c": [pattern_c_star],
        "patterns_a_star": [pattern_a],
        "patterns_b_star": [pattern_b],
        "patterns_c_star": [pattern_c],
    }
    return output_dict

# The way to do it would be
import numpy as np

from ..pattern_gen.mtp_tile import sample_tileset
from ..pattern_gen.mtp_main import sample_mtp_pattern, sample_default_mtp_pattern, NON_BORDERABLE, NON_FILLABLE, NO_EFFECT_TYPE, NON_BG
from ..pattern_gen.mtp_cellfx import sample_tile_ordering_fx
from ..pattern_gen.mtp_bgx import sample_bg, sample_border, default_border, sample_default_bg
from ..pattern_gen.mtp_layout import get_default_layout, sample_layout, sample_pre_deform, SINGLE_TILE_SPLITS, split_sampler_mapper, deform_sampler_mapper
from ..pattern_gen.splitting_func import split_default_mapper
from ..pattern_gen.filling_function import random_fill_rect_repeat, sample_fill, sample_repeat_fill
# Completely replace the layout
NON_DEFORMABLE = ["RandomFillRectRepeat", "RandomFillVoronoi"]
ROTATION_TYPES = ['RadialRepeat', 'RadialRepeatBricked', 'RadialRepeatFixedArc', 'RadialRepeatFixedArcBricked',]
TRANSLATION_TYPES = ['RectRepeat', 'RectRepeatFitting', 'OverlapX', 'OverlapY', 'OverlapXY', 'RectRepeatShiftedX', 'RectRepeatShiftedY', 'HexRepeat', 'HexRepeatY', 'VoronoiRepeat', 'IrregularRepeat']

def full_layout_change(filenames, filename_to_indices, harder_constraint=0, old_flag=False, *args, **kwargs):
    # Lets create 2 sets.
    # match n-tiles
    # Ensure not a random fill layout:
    found_pattern = False
    while not found_pattern:
        if harder_constraint in [1, 2, 3]: 
            # Match number of tiles, 
            # Match effects? 
            tileset = sample_tileset(filenames, filename_to_indices, old_mode=old_flag)
            n_tiles = len(tileset.tileset)
            pattern_a = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
            pattern_b = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
            if harder_constraint in [3, 4]:
                # match cellfx
                
                pattern_b.cellfx_cfg = pattern_a.cellfx_cfg.clone()

        else:
            pattern_a = sample_mtp_pattern(filenames, filename_to_indices)
            pattern_b = sample_mtp_pattern(filenames, filename_to_indices)
        a_type = pattern_a.layout_cfg.split._type
        b_type = pattern_b.layout_cfg.split._type
        # if a_type in NON_BG or b_type in NON_BG:
        #     found_pattern = False
        # else:
        found_pattern = True
        
    
    pattern_c = sample_default_mtp_pattern(filenames, filename_to_indices, n_tiles=1)
    # Just different tilesets.
    n_tiles_a = len(pattern_a.tile_cfg.tileset)
    n_tiles_b = len(pattern_b.tile_cfg.tileset)

    n_tiles = max(n_tiles_a, n_tiles_b)
    new_layout = sample_layout(n_tiles, old_mode=old_flag)
    split_type = new_layout.split._type

    pattern_a_star = pattern_a.clone()
    a_layout = new_layout.clone()
    if split_type == "RandomFillRectRepeat":
        # sample the random filler
        fill_cfg = random_fill_rect_repeat(pattern_a.tile_cfg, pattern_a.cellfx_cfg, split_type="RandomFillRectRepeat")
        a_layout.fill_cfg = fill_cfg
    elif split_type == "RandomFillVoronoi":
        fill_cfg = random_fill_rect_repeat(pattern_a.tile_cfg, pattern_a.cellfx_cfg, split_type="RandomFillVoronoi")
        a_layout.fill_cfg = fill_cfg
    pattern_a_star.layout_cfg = a_layout.clone()

    pattern_b_star = pattern_b.clone()
    b_layout = new_layout.clone()
    if split_type == "RandomFillRectRepeat":
        # sample the random filler
        fill_cfg = random_fill_rect_repeat(pattern_b.tile_cfg, pattern_b.cellfx_cfg, split_type="RandomFillRectRepeat")
        b_layout.fill_cfg = fill_cfg
    elif split_type == "RandomFillVoronoi":
        fill_cfg = random_fill_rect_repeat(pattern_b.tile_cfg, pattern_b.cellfx_cfg, split_type="RandomFillVoronoi")
        b_layout.fill_cfg = fill_cfg
    pattern_b_star.layout_cfg = b_layout.clone()

    pattern_c_star = pattern_c.clone()
    c_layout = new_layout.clone()
    if split_type == "RandomFillRectRepeat":
        # sample the random filler
        fill_cfg = random_fill_rect_repeat(pattern_c.tile_cfg, pattern_c.cellfx_cfg, split_type="RandomFillRectRepeat")
        c_layout.fill_cfg = fill_cfg
    elif split_type == "RandomFillVoronoi":
        fill_cfg = random_fill_rect_repeat(pattern_c.tile_cfg, pattern_c.cellfx_cfg, split_type="RandomFillVoronoi")
        c_layout.fill_cfg = fill_cfg
    pattern_c_star.layout_cfg = c_layout.clone()

    if split_type in NON_BORDERABLE: 
        pattern_a_star.border_cfg = None
        pattern_b_star.border_cfg = None
        pattern_c_star.border_cfg = None
    if split_type in NON_FILLABLE:
        pattern_a_star.fill_cfg = None
        pattern_b_star.fill_cfg = None
        pattern_c_star.fill_cfg = None

    if split_type in NO_EFFECT_TYPE:
        pattern_a_star.cellfx_cfg.effects = []
        pattern_b_star.cellfx_cfg.effects = []
        pattern_c_star.cellfx_cfg.effects = []

    output_dict = {
        "patterns_a": [pattern_a],
        "patterns_b": [pattern_b],
        "patterns_c": [pattern_c],
        "patterns_a_star": [pattern_a_star],
        "patterns_b_star": [pattern_b_star],
        "patterns_c_star": [pattern_c_star],
    }
    return output_dict

def split_change(filenames, filename_to_indices, old_flag=False, *args, **kwargs):
    # Lets create 2 sets.
    pattern_a = sample_mtp_pattern(filenames, filename_to_indices)
    pattern_b = sample_mtp_pattern(filenames, filename_to_indices)
    pattern_b.layout_cfg.pre_deform = pattern_a.layout_cfg.pre_deform.clone()
    pattern_c = sample_default_mtp_pattern(filenames, filename_to_indices, n_tiles=1)
    pattern_c.layout_cfg.pre_deform = pattern_a.layout_cfg.pre_deform.clone()
    # Just different tilesets.
    n_tiles_a = len(pattern_a.tile_cfg.tileset)
    n_tiles_b = len(pattern_b.tile_cfg.tileset)
    n_tiles = max(n_tiles_a, n_tiles_b)

    split_types = list(split_sampler_mapper.keys())
    if n_tiles > 1:
        split_types = [x for x in split_types if not x in SINGLE_TILE_SPLITS]
    if n_tiles > 2:
        remove_types = ["OverlapX", "OverlapY", "OverlapXY"]
        split_types = [x for x in split_types if not x in remove_types]
    # Simple_types:
    split, split_size = split_sampler_mapper[np.random.choice(split_types)](None, *args, **kwargs)
    # now 
    split_type = split._type

    pattern_a_star = pattern_a.clone()
    if split_type == "RandomFillRectRepeat":
        # sample the random filler
        fill_cfg = random_fill_rect_repeat(pattern_a.tile_cfg, pattern_a.cellfx_cfg, split_type="RandomFillRectRepeat")
        pattern_a_star.layout_cfg.fill_cfg = fill_cfg
    elif split_type == "RandomFillVoronoi":
        fill_cfg = random_fill_rect_repeat(pattern_a.tile_cfg, pattern_a.cellfx_cfg, split_type="RandomFillVoronoi")
        pattern_a_star.layout_cfg.fill_cfg = fill_cfg
    pattern_a_star.layout_cfg.split = split.clone()

    pattern_b_star = pattern_b.clone()
    if split_type == "RandomFillRectRepeat":
        # sample the random filler
        fill_cfg = random_fill_rect_repeat(pattern_b.tile_cfg, pattern_b.cellfx_cfg, split_type="RandomFillRectRepeat")
        pattern_b_star.layout_cfg.fill_cfg = fill_cfg
    elif split_type == "RandomFillVoronoi":
        fill_cfg = random_fill_rect_repeat(pattern_b.tile_cfg, pattern_b.cellfx_cfg, split_type="RandomFillVoronoi")
        pattern_b_star.layout_cfg.fill_cfg = fill_cfg
    pattern_b_star.layout_cfg.split = split.clone()

    pattern_c_star = pattern_c.clone()
    if split_type == "RandomFillRectRepeat":
        # sample the random filler
        fill_cfg = random_fill_rect_repeat(pattern_c.tile_cfg, pattern_c.cellfx_cfg, split_type="RandomFillRectRepeat")
        pattern_c_star.layout_cfg.fill_cfg = fill_cfg
    elif split_type == "RandomFillVoronoi":
        fill_cfg = random_fill_rect_repeat(pattern_c.tile_cfg, pattern_c.cellfx_cfg, split_type="RandomFillVoronoi")
        pattern_c_star.layout_cfg.fill_cfg = fill_cfg
    pattern_c_star.layout_cfg.split = split.clone()


    if split_type in NON_BORDERABLE: 
        pattern_a_star.border_cfg = None
        pattern_b_star.border_cfg = None
        pattern_c_star.border_cfg = None
    else:
        pattern_c_star.border_cfg = default_border(pattern_c.layout_cfg)
    if split_type in NON_FILLABLE:
        pattern_a_star.fill_cfg = None
        pattern_b_star.fill_cfg = None
        pattern_c_star.fill_cfg = None

    if split_type in NO_EFFECT_TYPE:
        pattern_a_star.cellfx_cfg.effects = []
        pattern_b_star.cellfx_cfg.effects = []
        pattern_c_star.cellfx_cfg.effects = []


    output_dict = {
        "patterns_a": [pattern_a],
        "patterns_b": [pattern_b],
        "patterns_c": [pattern_c],
        "patterns_a_star": [pattern_a_star],
        "patterns_b_star": [pattern_b_star],
        "patterns_c_star": [pattern_c_star],
    }
    return output_dict

### FAILED
def deform_change(filenames, filename_to_indices, old_flag=False, *args, **kwargs):
    # Lets create 2 sets.
    
    found_patterns = False
    while not found_patterns:
        pattern_a = sample_mtp_pattern(filenames, filename_to_indices)
        pattern_b = sample_mtp_pattern(filenames, filename_to_indices)
        a_type = pattern_a.layout_cfg.split._type
        b_type = pattern_b.layout_cfg.split._type
        if a_type in NON_DEFORMABLE or b_type in NON_DEFORMABLE:
            continue
        else:
            found_patterns = True
    pattern_c = sample_default_mtp_pattern(filenames, filename_to_indices, n_tiles=1)

    deform_types = list(deform_sampler_mapper.keys())
    deform_types.remove('no_deform')
    deform = deform_sampler_mapper[np.random.choice(deform_types)](None, *args, **kwargs)

    pattern_a_star = pattern_a.clone()
    pattern_a_star.layout_cfg.deform = deform.clone()

    pattern_b_star = pattern_b.clone()
    pattern_b_star.layout_cfg.deform = deform.clone()

    pattern_c_star = pattern_c.clone()
    pattern_c_star.layout_cfg.deform = deform.clone()

    # Just different tilesets.
    output_dict = {
        "patterns_a": [pattern_a],
        "patterns_b": [pattern_b],
        "patterns_c": [pattern_c],
        "patterns_a_star": [pattern_a_star],
        "patterns_b_star": [pattern_b_star],
        "patterns_c_star": [pattern_c_star],
    }
    return output_dict

# Rotations/Translations
def pre_deform_change(filenames, filename_to_indices, old_flag=False, *args, **kwargs):
    # Lets create 2 sets.
    set_made = False
    while not set_made:
        pattern_a = sample_mtp_pattern(filenames, filename_to_indices)
        a_type = pattern_a.layout_cfg.split._type
        if a_type in ROTATION_TYPES:
            not_found = True
            while not_found:
                pattern_b = sample_mtp_pattern(filenames, filename_to_indices)
                b_type = pattern_b.layout_cfg.split._type
                if b_type in ROTATION_TYPES:
                    not_found = False
                    set_made = True
                    set_type = "rotation"
        elif a_type in TRANSLATION_TYPES:
            not_found = True
            while not_found:
                pattern_b = sample_mtp_pattern(filenames, filename_to_indices)
                b_type = pattern_b.layout_cfg.split._type
                if b_type in TRANSLATION_TYPES:
                    not_found = False
                    set_made=True
                    set_type = "translation"
        
    pattern_c = sample_default_mtp_pattern(filenames, filename_to_indices, n_tiles=1)
    
    pattern_b.layout_cfg.pre_deform = pattern_a.layout_cfg.pre_deform.clone()
    if set_type == "rotation":
        # Means I can move the center.
        pattern_c.layout_cfg.split = split_default_mapper["RadialRepeatFixedArc"](None, *args, **kwargs)
        new_x = np.random.uniform(-1, 1)
        new_y = np.random.uniform(-1, 1)
        new_x = float(new_x)
        new_y = float(new_y)
        pattern_a_star = pattern_a.clone()
        pattern_a_star.layout_cfg.split.center_x = new_x
        pattern_a_star.layout_cfg.split.center_y = new_y
        pattern_b_star = pattern_b.clone()
        pattern_b_star.layout_cfg.split.center_x = new_x
        pattern_b_star.layout_cfg.split.center_y = new_y
        pattern_c_star = pattern_c.clone()
        pattern_c_star.layout_cfg.split.center_x = new_x
        pattern_c_star.layout_cfg.split.center_y = new_y
    
    else:
        # pattern_c.layout_cfg.pre_deform = pattern_a.layout_cfg.pre_deform.clone()
        # This means we can control init rotation.
        new_theta = np.random.uniform(-np.pi/4 , np.pi/4)
        new_theta = float(new_theta)
        pattern_a_star = pattern_a.clone()
        pattern_a_star.layout_cfg.pre_deform.rot = new_theta
        pattern_b_star = pattern_b.clone()
        pattern_b_star.layout_cfg.pre_deform.rot = new_theta
        pattern_c_star = pattern_c.clone()
        pattern_c_star.layout_cfg.pre_deform.rot = new_theta
    # Just different tilesets.
    output_dict = {
        "patterns_a": [pattern_a],
        "patterns_b": [pattern_b],
        "patterns_c": [pattern_c],
        "patterns_a_star": [pattern_a_star],
        "patterns_b_star": [pattern_b_star],
        "patterns_c_star": [pattern_c_star],
    }
    return output_dict
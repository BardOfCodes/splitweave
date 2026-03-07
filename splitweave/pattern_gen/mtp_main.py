
# DO BG border etc later
# for now 
import os
from collections import defaultdict
from yacs.config import CfgNode as CN
import numpy as np

from .mtp_layout import sample_layout, get_default_layout
from .mtp_tile import sample_tileset, get_default_tileset
from .mtp_cellfx import sample_mtp_cellfx, default_mtp_cellfx
from .mtp_bgx import sample_bg, sample_border, sample_default_bg, default_border
from .filling_function import random_fill_rect_repeat, sample_fill, sample_repeat_fill

NO_EFFECT_TYPE = [
    "OverlapX", "OverlapY", "OverlapXY" # Now the question is where the tile info comes in. I think TileSet it is. 
]
PARTIAL_EFFECT_TYPE = [
    "RectRepeatFitting"
]
NON_BORDERABLE = [
    "OverlapX", "OverlapY", "OverlapXY",
    "RandomFillRectRepeat", "RandomFillVoronoi"
]
NON_FILLABLE = [
    "OverlapX", "OverlapY", "OverlapXY",
    "RandomFillRectRepeat", "RandomFillVoronoi",
    "VoronoiRepeat", "IrregularRepeat",
    "RadialRepeat", "RadialRepeatBricked", 
    "RadialRepeatFixedArc", "RadialRepeatFixedArcBricked",
]
NON_BG = [
    "RandomFillRectRepeat", "RandomFillVoronoi",

]


def sample_mtp_pattern(filenames, filename_to_indices, *args, **kwargs):
    cfg = CN()
    if "tile_cfg" in kwargs:
        tile_cfg = kwargs["tile_cfg"]
    else:
        tile_cfg = sample_tileset(filenames, filename_to_indices, *args, **kwargs)
    n_tiles = len(tile_cfg.tileset)

    if not "n_tiles" in kwargs:
        kwargs["n_tiles"] = n_tiles 
    if "layout_cfg" in kwargs:
        layout_cfg = kwargs["layout_cfg"]
    else:
        layout_cfg = sample_layout(*args, **kwargs)

    split_type = layout_cfg.split._type
    # In MAIN EVAL we need to take care of the alternative execution models.
    if "cellfx_cfg" in kwargs:
        cellfx_cfg = kwargs["cellfx_cfg"]
    else:
        if split_type in NO_EFFECT_TYPE:
            if "n_tiles" in kwargs:
                n_tiles = kwargs["n_tiles"]
            else:
                if layout_cfg.split.multi_tile:
                    n_tiles = 2
                else:
                    n_tiles = 1
            cellfx_cfg = default_mtp_cellfx(n_tiles=n_tiles)
        else:
            partial_effects = split_type in PARTIAL_EFFECT_TYPE
            if np.random.choice([True, False]):
                cellfx_cfg = sample_mtp_cellfx(n_tiles=n_tiles, partial_effects=partial_effects)
            else:
                cellfx_cfg = default_mtp_cellfx(n_tiles=n_tiles)
    
    if split_type == "RandomFillRectRepeat":
        # sample the random filler
        fill_cfg = random_fill_rect_repeat(tile_cfg, cellfx_cfg, split_type="RandomFillRectRepeat")
        layout_cfg.fill_cfg = fill_cfg
    elif split_type == "RandomFillVoronoi":
        fill_cfg = random_fill_rect_repeat(tile_cfg, cellfx_cfg, split_type="RandomFillVoronoi")
        layout_cfg.fill_cfg = fill_cfg
    # Also get overlap and other working here. 
    if "bg_cfg" in kwargs:
        bg_cfg = kwargs["bg_cfg"]
    else:
        bg_cfg = sample_bg()
    

    if not split_type in NON_BORDERABLE: 
        do_border = np.random.choice([True, False])
    else:
        do_border = False

    if (not split_type in NON_FILLABLE) and (bg_cfg.bg_mode == "plain") and (not do_border):
        do_fill = np.random.choice([True, False])
    else:
        do_fill = False


    if do_border:
        if "border_cfg" in kwargs:
            border_cfg = kwargs["border_cfg"]
        else:
            border_cfg = sample_border(layout_cfg)
    else:
        border_cfg = None
    if do_fill:
        # Have the brick style and random mode. 
        if "fill_cfg" in kwargs:
            fill_cfg = kwargs["fill_cfg"]
        else:
            if np.random.choice([True, False]):
                fill_cfg = sample_fill(filenames, filename_to_indices, *args, **kwargs)
            else:
                fill_cfg = sample_repeat_fill(filenames, filename_to_indices, *args, **kwargs)
    else:
        fill_cfg = None
    # need to set the seed
    # fill_cfg = sample_repeat_fill(filenames, filename_to_indices, *args, **kwargs)
    cfg.bg_cfg = bg_cfg
    cfg.border_cfg = border_cfg
    cfg.fill_cfg = fill_cfg
    cfg.tile_cfg = tile_cfg
    cfg.layout_cfg = layout_cfg
    cfg.cellfx_cfg = cellfx_cfg
    return cfg


def sample_default_mtp_pattern(filenames, filename_to_indices, *args, **kwargs):
    cfg = CN()
    if "tile_cfg" in kwargs:
        tile_cfg = kwargs["tile_cfg"]
    else:
        tile_cfg = sample_tileset(filenames, filename_to_indices, *args, **kwargs)
    n_tiles = len(tile_cfg.tileset)

    if not "n_tiles" in kwargs:
        kwargs["n_tiles"] = n_tiles 
    if "layout_cfg" in kwargs:
        layout_cfg = kwargs["layout_cfg"]
    else:
        layout_cfg = get_default_layout(n_tiles, *args, **kwargs)

    split_type = layout_cfg.split._type
    # In MAIN EVAL we need to take care of the alternative execution models.
    if "cellfx_cfg" in kwargs:
        cellfx_cfg = kwargs["cellfx_cfg"]
    else:
        cellfx_cfg = default_mtp_cellfx(n_tiles=n_tiles)
    
    if split_type == "RandomFillRectRepeat":
        # sample the random filler
        fill_cfg = random_fill_rect_repeat(tile_cfg, cellfx_cfg, split_type="RandomFillRectRepeat")
        layout_cfg.fill_cfg = fill_cfg
    elif split_type == "RandomFillVoronoi":
        fill_cfg = random_fill_rect_repeat(tile_cfg, cellfx_cfg, split_type="RandomFillVoronoi")
        layout_cfg.fill_cfg = fill_cfg
    # Also get overlap and other working here. 
    if "bg_cfg" in kwargs:
        bg_cfg = kwargs["bg_cfg"]
    else:
        bg_cfg = sample_default_bg()
    

    if "border_cfg" in kwargs:
        border_cfg = kwargs["border_cfg"]
    else:
        border_cfg = default_border(layout_cfg)
    if "fill_cfg" in kwargs:
        fill_cfg = kwargs["fill_cfg"]
    else:
        fill_cfg = None
    # fill_cfg = sample_repeat_fill(filenames, filename_to_indices, *args, **kwargs)
    cfg.bg_cfg = bg_cfg
    cfg.border_cfg = border_cfg
    cfg.fill_cfg = fill_cfg
    cfg.tile_cfg = tile_cfg
    cfg.layout_cfg = layout_cfg
    cfg.cellfx_cfg = cellfx_cfg
    return cfg
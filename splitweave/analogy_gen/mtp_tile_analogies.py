# The way to do it would be
import numpy as np

from ..pattern_gen.mtp_tile import sample_tileset
from ..pattern_gen.mtp_main import sample_mtp_pattern, sample_default_mtp_pattern
from ..pattern_gen.mtp_cellfx import sample_tile_ordering_fx
from ..pattern_gen.mtp_layout import META_SPLIT_MAPPER

SINGLE_TILE_SPLITS = [
    "RectRepeatFitting", "RandomFillRectRepeat", "RandomFillVoronoi",
]
TWO_TILE_SPLITS = ["OverlapX", "OverlapY", "OverlapXY"]

def full_tileset_change(filenames, filename_to_indices, harder_constraint=0, n_tiles=None, old_flag=False, *args, **kwargs):
    
    # Lets create 2 sets.
    if harder_constraint == 1:
        if not n_tiles == None:
            tileset = sample_tileset(filenames, filename_to_indices, n_tiles=n_tiles, old_mode=old_flag)
        else:
            tileset = sample_tileset(filenames, filename_to_indices, old_mode=old_flag)
        n_tiles = len(tileset.tileset)
        # Also force the layout to have the same meta class
        found_pattern = False
        resample_both = True
        while not found_pattern:
            if resample_both:
                pattern_a = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
                pattern_b = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
            else:
                pattern_b = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
            a_type = pattern_a.layout_cfg.split._type
            b_type = pattern_b.layout_cfg.split._type
            meta_split_a = META_SPLIT_MAPPER[a_type]
            meta_split_b = META_SPLIT_MAPPER[b_type]
            if meta_split_a == meta_split_b:
                found_pattern = True
            else:
                found_pattern = False
                resample_both = False # keep probability same
    elif harder_constraint == 2:

        tileset = sample_tileset(filenames, filename_to_indices, old_mode=old_flag)
        n_tiles = len(tileset.tileset)
        # Also force the layout to have the same meta class
        found_pattern = False
        resample_both = True
        while not found_pattern:
            if resample_both:
                pattern_a = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
                pattern_b = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
            else:
                pattern_b = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
            a_type = pattern_a.layout_cfg.split._type
            b_type = pattern_b.layout_cfg.split._type
            meta_split_a = a_type
            meta_split_b = b_type
            if meta_split_a == meta_split_b:
                found_pattern = True
            else:
                found_pattern = False
                resample_both = False # keep probability same
    elif harder_constraint == 3:
        pattern_a = sample_mtp_pattern(filenames, filename_to_indices)
        pattern_b = sample_mtp_pattern(filenames, filename_to_indices)
        pattern_b.layout_cfg = pattern_a.layout_cfg.clone()
        # 
    elif harder_constraint == 4:
        # should the tile order also match?
        pattern_a = sample_mtp_pattern(filenames, filename_to_indices)
        pattern_b = sample_mtp_pattern(filenames, filename_to_indices)
        pattern_b.layout_cfg = pattern_a.layout_cfg.clone()
        n_tiles_a = len(pattern_a.tile_cfg.tileset)
        n_tiles_b = len(pattern_b.tile_cfg.tileset)
        if n_tiles_a == n_tiles_b:
            pattern_b.cellfx_cfg.tile_order = pattern_a.cellfx_cfg.tile_order.clone()
    else:
        pattern_a = sample_mtp_pattern(filenames, filename_to_indices)
        pattern_b = sample_mtp_pattern(filenames, filename_to_indices)

    pattern_c = sample_default_mtp_pattern(filenames, filename_to_indices, n_tiles=1)
    # Just different tilesets.

    splits_a = pattern_a.layout_cfg.split._type
    splits_b = pattern_b.layout_cfg.split._type
    splits = [splits_a, splits_b]
    new_n_tiles = None
    if any([split in SINGLE_TILE_SPLITS for split in splits]):
        new_n_tiles = 1
    if any([split in TWO_TILE_SPLITS for split in splits]):
        new_n_tiles = int(np.random.choice([1, 2]))

    tileset = sample_tileset(filenames, filename_to_indices, n_tiles=new_n_tiles, old_mode=old_flag)
    new_tile_order = sample_tile_ordering_fx(None, n_tiles=len(tileset.tileset))



    pattern_a_star = pattern_a.clone()
    pattern_a_star.tile_cfg = tileset.clone()
    if pattern_a_star.cellfx_cfg is not None:
        pattern_a_star.cellfx_cfg.tile_order = new_tile_order.clone()
    pattern_b_star = pattern_b.clone()
    pattern_b_star.tile_cfg = tileset.clone()
    if pattern_b_star.cellfx_cfg is not None:
        pattern_b_star.cellfx_cfg.tile_order = new_tile_order.clone()
    pattern_c_star = pattern_c.clone()
    pattern_c_star.tile_cfg = tileset.clone()
    if pattern_c_star.cellfx_cfg is not None:
        pattern_c_star.cellfx_cfg.tile_order = new_tile_order.clone()
    # To ensure that the tile order is correct, it should be matched. 
    # Pattern C required as well. 
    output_dict = {
        "patterns_a": [pattern_a],
        "patterns_b": [pattern_b],
        "patterns_c": [pattern_c],
        "patterns_a_star": [pattern_a_star],
        "patterns_b_star": [pattern_b_star],
        "patterns_c_star": [pattern_c_star],
    }
    return output_dict

def partial_tileset_change(filenames, filename_to_indices, harder_constraint=0, old_flag=False, *args, **kwargs):

    if harder_constraint==1:
        n_tiles = int(np.random.choice([1, 2, 3]))
        n_tiles_a = n_tiles
        n_tiles_b = n_tiles
        found_pattern = False
        resample_both = True
        while not found_pattern:
            if resample_both:
                pattern_a = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
                pattern_b = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
            else:
                pattern_b = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
            pattern_b.cellfx_cfg.tile_order = pattern_a.cellfx_cfg.tile_order.clone()
            a_type = pattern_a.layout_cfg.split._type
            b_type = pattern_b.layout_cfg.split._type
            meta_split_a = META_SPLIT_MAPPER[a_type]
            meta_split_b = META_SPLIT_MAPPER[b_type]
            if meta_split_a == meta_split_b:
                found_pattern = True
            else:
                found_pattern = False
                resample_both = False # keep probability same
    elif harder_constraint == 2:

        n_tiles = int(np.random.choice([1, 2, 3]))
        n_tiles_a = n_tiles
        n_tiles_b = n_tiles
        found_pattern = False
        resample_both = True
        while not found_pattern:
            if resample_both:
                pattern_a = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
                pattern_b = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
            else:
                pattern_b = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
            pattern_b.cellfx_cfg.tile_order = pattern_a.cellfx_cfg.tile_order.clone()
            a_type = pattern_a.layout_cfg.split._type
            b_type = pattern_b.layout_cfg.split._type
            meta_split_a = a_type
            meta_split_b = b_type
            if meta_split_a == meta_split_b:
                found_pattern = True
            else:
                found_pattern = False
                resample_both = False # keep probability same
    elif harder_constraint == 3:

        n_tiles = int(np.random.choice([1, 2, 3]))
        n_tiles_a = n_tiles
        n_tiles_b = n_tiles
        pattern_a = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
        pattern_b = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
        pattern_b.layout_cfg = pattern_a.layout_cfg.clone()
    elif harder_constraint == 4:
        # should the tile order also match?
        n_tiles = int(np.random.choice([1, 2, 3]))
        n_tiles_a = n_tiles
        n_tiles_b = n_tiles
        pattern_a = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
        pattern_b = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
        pattern_b.layout_cfg = pattern_a.layout_cfg.clone()
        n_tiles_a = len(pattern_a.tile_cfg.tileset)
        n_tiles_b = len(pattern_b.tile_cfg.tileset)
        if n_tiles_a == n_tiles_b:
            pattern_b.cellfx_cfg.tile_order = pattern_a.cellfx_cfg.tile_order.clone()
    else:
        n_tiles_a = int(np.random.choice([1, 2, 3]))
        pattern_a = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles_a)
        n_tiles_b = int(np.random.choice([1, 2, 3]))
        pattern_b = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles_b)

    rand_a = np.random.choice(n_tiles_a)
    rand_b = np.random.choice(n_tiles_b)
    # n_tiles = int(np.random.choice([1, 2, 3]))
    # int(np.random.choice([1, 2,]))
    if n_tiles_a > 1 or n_tiles_b > 1:
        n_tiles_c = 2
    else:
        n_tiles_c = 1
    pattern_c = sample_default_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles_c)
    rand_c = np.random.choice(n_tiles_c)
    tileset = []
    for i in range(2):
        cur_tile = sample_tileset(filenames, filename_to_indices, n_tiles=1, old_mode=old_flag)
        cur_tile = cur_tile.tileset[0]
        tileset.append(cur_tile)

    pattern_a_star = pattern_a.clone()
    pattern_b_star = pattern_b.clone()
    pattern_c_star = pattern_c.clone()
    pattern_a.tile_cfg.tileset[rand_a] = tileset[0].clone()
    pattern_b.tile_cfg.tileset[rand_b] = tileset[0].clone()
    pattern_c.tile_cfg.tileset[rand_c] = tileset[0].clone()
    pattern_a_star.tile_cfg.tileset[rand_a] = tileset[0 + 1].clone()
    pattern_b_star.tile_cfg.tileset[rand_b] = tileset[0 + 1].clone()
    pattern_c_star.tile_cfg.tileset[rand_c] = tileset[0 + 1]
    # For partial change, we need to match some of the tiles and replace them. 
    output_dict = {
        "patterns_a": [pattern_a],
        "patterns_b": [pattern_b],
        "patterns_c": [pattern_c],
        "patterns_a_star": [pattern_a_star],
        "patterns_b_star": [pattern_b_star],
        "patterns_c_star": [pattern_c_star],
    }
    return output_dict

def remove_n_tiles(filenames, filename_to_indices, harder_constraint=0, old_flag=False, *args, **kwargs):

    # there should be at least 2 tiles.
    # delete a shared tile. 
    # update the tile-order. 
    n_tiles = int(np.random.choice([2, 3]))
    if harder_constraint == 1: 
        found_pattern = False
        resample_both = True
        while not found_pattern:
            if resample_both:
                pattern_a = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
                pattern_b = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
            else:
                pattern_b = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
            a_type = pattern_a.layout_cfg.split._type
            b_type = pattern_b.layout_cfg.split._type
            meta_split_a = META_SPLIT_MAPPER[a_type]
            meta_split_b = META_SPLIT_MAPPER[b_type]
            if meta_split_a == meta_split_b:
                found_pattern = True
            else:
                found_pattern = False
                resample_both = False # keep probability same
    elif harder_constraint == 2:
        found_pattern = False
        resample_both = True
        while not found_pattern:
            if resample_both:
                pattern_a = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
                pattern_b = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
            else:
                pattern_b = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
            a_type = pattern_a.layout_cfg.split._type
            b_type = pattern_b.layout_cfg.split._type
            meta_split_a = a_type
            meta_split_b = b_type
            if meta_split_a == meta_split_b:
                found_pattern = True
            else:
                found_pattern = False
                resample_both = False # keep probability same

    elif harder_constraint == 3:

        n_tiles = int(np.random.choice([2, 3, 4]))
        n_tiles_a = n_tiles
        n_tiles_b = n_tiles
        pattern_a = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
        pattern_b = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
        pattern_b.layout_cfg = pattern_a.layout_cfg.clone()
    elif harder_constraint == 4:
        # should the tile order also match?
        n_tiles = int(np.random.choice([2, 3, 4]))
        n_tiles_a = n_tiles
        n_tiles_b = n_tiles
        pattern_a = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
        pattern_b = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
        pattern_b.layout_cfg = pattern_a.layout_cfg.clone()
        pattern_b.cellfx_cfg.tile_order = pattern_a.cellfx_cfg.tile_order.clone()
    else:
        pattern_a = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
        # n_tiles = int(np.random.choice([2, 3, 4]))
        pattern_b = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)


    rand_a = np.random.choice(n_tiles)
    rand_b = np.random.choice(n_tiles)
    # n_tiles = int(np.random.choice([2]))
    pattern_c = sample_default_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
    rand_c = np.random.choice(n_tiles)
    
    # Select the tile that is going to be deleted
    delete_tile = sample_tileset(filenames, filename_to_indices, n_tiles=1, old_mode=old_flag)
    
    # delete a shared tile.
    pattern_a.tile_cfg.tileset[rand_a] = delete_tile.tileset[0].clone()
    pattern_b.tile_cfg.tileset[rand_b] = delete_tile.tileset[0].clone()
    pattern_c.tile_cfg.tileset[rand_c] = delete_tile.tileset[0].clone()
    pattern_a_star = pattern_a.clone()
    pattern_b_star = pattern_b.clone()
    pattern_c_star = pattern_c.clone()

    pattern_a_star.tile_cfg.tileset = [x for x in pattern_a.tile_cfg.tileset]
    pattern_b_star.tile_cfg.tileset = [x for x in pattern_b.tile_cfg.tileset]
    pattern_c_star.tile_cfg.tileset = [x for x in pattern_c.tile_cfg.tileset]

    pattern_a_star.tile_cfg.tileset.pop(rand_a)
    if len(pattern_a_star.tile_cfg.tileset) == 1:
        pattern_a_star.cellfx_cfg.tile_order = sample_tile_ordering_fx(None, n_tiles=len(pattern_a_star.tile_cfg.tileset))
    else:
        pattern_a_star.cellfx_cfg.tile_order.signal.k -= 1
        pattern_a_star.cellfx_cfg.tile_order.n_tiles -= 1

    pattern_b_star.tile_cfg.tileset.pop(rand_b)
    if len(pattern_b_star.tile_cfg.tileset) == 1:
        pattern_b_star.cellfx_cfg.tile_order = sample_tile_ordering_fx(None, n_tiles=len(pattern_b_star.tile_cfg.tileset))
    else:
        pattern_b_star.cellfx_cfg.tile_order.signal.k -= 1
        pattern_b_star.cellfx_cfg.tile_order.n_tiles -= 1

    # if len(pattern_b_star.tile_cfg.tileset) == len(pattern_a_star.tile_cfg.tileset):
    #     pattern_b_star.cellfx_cfg.tile_order = pattern_a_star.cellfx_cfg.tile_order.clone()
    # else:
    #     pattern_b_star.cellfx_cfg.tile_order = sample_tile_ordering_fx(None, n_tiles=len(pattern_b_star.tile_cfg.tileset))
    
    pattern_c_star.tile_cfg.tileset.pop(rand_c)
    if len(pattern_c_star.tile_cfg.tileset) == 1:
        pattern_c_star.cellfx_cfg.tile_order = sample_tile_ordering_fx(None, n_tiles=len(pattern_c_star.tile_cfg.tileset))
    else:
        pattern_c_star.cellfx_cfg.tile_order.signal.k -= 1
        pattern_c_star.cellfx_cfg.tile_order.n_tiles -= 1
    # if len(pattern_c_star.tile_cfg.tileset) == len(pattern_a_star.tile_cfg.tileset):
    #     pattern_c_star.cellfx_cfg.tile_order = pattern_a_star.cellfx_cfg.tile_order.clone()
    # else:
    #     pattern_c_star.cellfx_cfg.tile_order = sample_tile_ordering_fx(None, n_tiles=len(pattern_c_star.tile_cfg.tileset))

    output_dict = {
        "patterns_a": [pattern_a],
        "patterns_b": [pattern_b],
        "patterns_c": [pattern_c],
        "patterns_a_star": [pattern_a_star],
        "patterns_b_star": [pattern_b_star],
        "patterns_c_star": [pattern_c_star],
    }
    return output_dict

def add_n_tiles(filenames, filename_to_indices, harder_constraint=0, old_flag=False, *args, **kwargs):

    # there should be less than 3 tiles.
    # add a tile. 
    # update the tile-order.
    add_tile = sample_tileset(filenames, filename_to_indices, n_tiles=1, old_mode=old_flag)
    add_tile = add_tile.tileset[0]
    n_tiles = int(np.random.choice([1, 2]))

    if harder_constraint==1: 
        found_pattern = False
        resample_both = True
        while not found_pattern:
            if resample_both:
                pattern_a = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
                pattern_b = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
            else:
                pattern_b = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
            a_type = pattern_a.layout_cfg.split._type
            b_type = pattern_b.layout_cfg.split._type
            meta_split_a = META_SPLIT_MAPPER[a_type]
            meta_split_b = META_SPLIT_MAPPER[b_type]
            if meta_split_a == meta_split_b:
                found_pattern = True
            else:
                found_pattern = False
                resample_both = False # keep probability same
    elif harder_constraint == 2:
        found_pattern = False
        resample_both = True
        while not found_pattern:
            if resample_both:
                pattern_a = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
                pattern_b = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
            else:
                pattern_b = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
            a_type = pattern_a.layout_cfg.split._type
            b_type = pattern_b.layout_cfg.split._type
            meta_split_a = a_type
            meta_split_b = b_type
            if meta_split_a == meta_split_b:
                found_pattern = True
            else:
                found_pattern = False
                resample_both = False # keep probability same
    elif harder_constraint == 4:
        # should the tile order also match?
        n_tiles = int(np.random.choice([1, 2, 3]))
        n_tiles_a = n_tiles
        n_tiles_b = n_tiles
        pattern_a = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
        pattern_b = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
        pattern_b.layout_cfg = pattern_a.layout_cfg.clone()
        pattern_b.cellfx_cfg.tile_order = pattern_a.cellfx_cfg.tile_order.clone()
    else:
        pattern_a = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)
        # n_tiles = int(np.random.choice([2, 3, 4]))
        pattern_b = sample_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)

    # n_tiles = int(np.random.choice([1,]))
    pattern_c = sample_default_mtp_pattern(filenames, filename_to_indices, n_tiles=n_tiles)

    pattern_a_star = pattern_a.clone()
    pattern_b_star = pattern_b.clone()
    pattern_c_star = pattern_c.clone()
    
    pattern_a_star.tile_cfg.tileset = [x for x in pattern_a.tile_cfg.tileset]
    pattern_b_star.tile_cfg.tileset = [x for x in pattern_b.tile_cfg.tileset]
    pattern_c_star.tile_cfg.tileset = [x for x in pattern_c.tile_cfg.tileset]

    pattern_a_star.tile_cfg.tileset.append(add_tile)
    if not "signal" in pattern_b.cellfx_cfg.tile_order:
        pattern_a_star.cellfx_cfg.tile_order = sample_tile_ordering_fx(None, n_tiles=len(pattern_a_star.tile_cfg.tileset))
    else:
        pattern_a_star.cellfx_cfg.tile_order.signal.k += 1
        pattern_a_star.cellfx_cfg.tile_order.n_tiles += 1

    n_tiles_a = len(pattern_a.tile_cfg.tileset)
    n_tiles_b = len(pattern_b.tile_cfg.tileset)
    
    pattern_b_star.tile_cfg.tileset.append(add_tile)
    if not "signal" in pattern_b.cellfx_cfg.tile_order:
        if n_tiles_a == n_tiles_b:
            pattern_b_star.cellfx_cfg.tile_order = pattern_a_star.cellfx_cfg.tile_order.clone()
        else:
            pattern_b_star.cellfx_cfg.tile_order = sample_tile_ordering_fx(None, n_tiles=len(pattern_b_star.tile_cfg.tileset))
    else:
        pattern_b_star.cellfx_cfg.tile_order.signal.k += 1
        pattern_b_star.cellfx_cfg.tile_order.n_tiles += 1

    pattern_c_star.tile_cfg.tileset.append(add_tile)
    if not "signal" in pattern_c.cellfx_cfg.tile_order:
        pattern_c_star.cellfx_cfg.tile_order = sample_tile_ordering_fx(None, n_tiles=len(pattern_c_star.tile_cfg.tileset))
    else:
        pattern_c_star.cellfx_cfg.tile_order.signal.k += 1
        pattern_c_star.cellfx_cfg.tile_order.n_tiles += 1
        
    output_dict = {
        "patterns_a": [pattern_a],
        "patterns_b": [pattern_b],
        "patterns_c": [pattern_c],
        "patterns_a_star": [pattern_a_star],
        "patterns_b_star": [pattern_b_star],
        "patterns_c_star": [pattern_c_star],
    }
    return output_dict
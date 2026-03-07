# Tile Bank -> Load 4 tiles. 
# For each tile # "cell Effects" " 50% no effects"
import numpy as np
from yacs.config import CfgNode as CN
from .utils import add_to_registry

tile_sampler_mapper = {}
tile_variable_mapper = {}
tile_default_mapper = {}

# Tile - Tile NAME, Tile FX
# Tile FX - rotate, scale, flip, recolor, outline, shadow, opacity
BASE_TILES = ""
OUTLINE_MIN = 0.01
OUTLINE_MAX = 0.05
SHADOW_MIN = 0.01
SHADOW_MAX = 0.05

def replace_tile_names(cfg1, filename_to_indices):
    
    tile_name = "_".join(cfg1.tilefile.split("_")[:-1])
    cur_index = int(cfg1.tilefile.split("_")[-1].split(".")[0])
    valid_indices = filename_to_indices[tile_name]
    if valid_indices.index(cur_index) < len(valid_indices) -1:
        new_index = valid_indices[valid_indices.index(cur_index) + 1]
    else:
        if valid_indices.index(cur_index) > 1:
            new_index = valid_indices[valid_indices.index(cur_index) - 1]
        else:
            new_index = np.random.choice(valid_indices)
    tilefile = f"{tile_name}_{new_index}.png"
    return tilefile

def get_name_set_real(all_filenames, filename_to_indices, k=4):
    # composed_all_filenames = []
    
    # for filename in all_filenames:
    #     file_inds = filename_to_indices[filename]
    #     for ind in file_inds:
    #         composed_all_filenames.append(f"{filename}_{ind}.png")
    # filenames = []
    filename = np.random.choice(all_filenames)
    file_inds = filename_to_indices[filename]

    # return composed_all_filenames
    # get a sequence of k consecutive indices
    ind = np.random.choice(len(file_inds) - k)
    n = max(file_inds) // k
    count = 0
    fail = False
    while True:
        ind = np.random.choice(n) * k
        all_in = True
        for i in range(k):
            if ind + i not in file_inds:
                all_in = False
                break
        if all_in:
            break
        count += 1
        if count == k:
            fail = True
            break
    if fail:
        ind = np.random.choice(len(file_inds) - k)
        new_name_set = [f"{filename}_{file_inds[ind + i]}.png" for i in range(k)]
    else:
        new_name_set = [f"{filename}_{ind + i}.png" for i in range(k)]
    return new_name_set


def get_name_set(all_filenames, filename_to_indices, k=4):
    # composed_all_filenames = []
    
    # for filename in all_filenames:
    #     file_inds = filename_to_indices[filename]
    #     for ind in file_inds:
    #         composed_all_filenames.append(f"{filename}_{ind}.png")
    # filenames = []
    random_names = np.random.choice(all_filenames, size=len(all_filenames), replace=False)
    new_name_set = [f"{filename}_{ind}.png" for filename in random_names for ind in filename_to_indices[filename]]
    return new_name_set

@add_to_registry(tile_sampler_mapper, "tileset")
def sample_tileset(all_filenames, filename_to_indices, 
                   overlap_only=False, n_tiles=None, 
                   old_mode=False, *args, **kwargs):
    cfg = CN()
    # variations: 
    # Same tile with variations.
    # Replace a certain tile. 
    # replace a certain tile effects
    # replace all tiles. 
    
    # potentially different
    tile_one = sample_tile(all_filenames, filename_to_indices, old_mode=old_mode, *args, **kwargs)
    tile_two = sample_tile(all_filenames, filename_to_indices, old_mode=old_mode, *args, **kwargs)
    tile_three = sample_tile(all_filenames, filename_to_indices, old_mode=old_mode, *args, **kwargs)
    tile_four = sample_tile(all_filenames, filename_to_indices, old_mode=old_mode, *args, **kwargs)
    # Get a sequence of four names, and replace them.
    if not old_mode:
        new_name_set = get_name_set(all_filenames, filename_to_indices, k=4)
        tile_one.tilefile = new_name_set[0]
        tile_two.tilefile = new_name_set[1]
        tile_three.tilefile = new_name_set[2]
        tile_four.tilefile = new_name_set[3]
    tileset = [tile_one, tile_two, tile_three, tile_four]

    if overlap_only:
        n_tiles = int(np.random.choice([1, 2]))

    if n_tiles is None:
        n_tiles = int(np.random.choice([1, 2, 3, 4], p=[0.75, 0.13, 0.08, 0.04]))
    else:
        n_tiles = int(n_tiles)
    
    tile_order = list(np.random.choice([0, 1, 2, 3], n_tiles, replace=False))
    sel_tiles = [tileset[x] for x in tile_order]
    cfg.tileset = sel_tiles
    return cfg

@add_to_registry(tile_variable_mapper, "tileset")
def var_resampler_tileset(current_cfg, all_filenames, filename_to_indices, *args, **kwargs):
    resamplers = {}
    # THe functions are
    # Do it only if active
    active_index = get_active_index(current_cfg)
    labels = ["tile_one", "tile_two", "tile_three", "tile_four"]
    for i, label in enumerate(labels):
        if active_index[i]:
            cur = var_resampler_tile(current_cfg[label], all_filenames, filename_to_indices, *args, **kwargs)
            cur_with_updated_keys = {f"{label}.{k}": v for k, v in cur.items()}
            resamplers.update(cur_with_updated_keys)
        else:
            resamplers[label] = None
    return resamplers

@add_to_registry(tile_default_mapper, "tileset")
def get_default_tileset(*args, **kwargs):
    cfg = CN()
    tile_one = get_default_tile(index=0, *args, **kwargs)
    tile_two = get_default_tile(index=1, *args, **kwargs)
    tile_three = get_default_tile(index=2, *args, **kwargs)
    tile_four = get_default_tile(index=3, *args, **kwargs)
    cfg.tileset = [tile_one, tile_two, tile_three, tile_four]
    return cfg

@add_to_registry(tile_sampler_mapper, "retro_tile")
def sample_retro_tile(all_filenames, filename_to_indices, old_mode=False, *args, **kwargs):
    cfg = CN()
    if old_mode:
        filename = np.random.choice(all_filenames)
        cfg.tilefile = f"{filename}"
    else:
        filename = np.random.choice(all_filenames)
        file_ind = np.random.choice(filename_to_indices[filename])
        cfg.tilefile = f"{filename}_{file_ind}.png"
    return cfg

@add_to_registry(tile_default_mapper, "retro_tile")
def default_retro_tile(*args, **kwargs):
    cfg = CN()
    index = kwargs.get("index", 0)
    cfg.tilefile = BASE_TILES[index]
    return cfg

@add_to_registry(tile_sampler_mapper, "tile")
def sample_tile(all_filenames, filename_to_indices, enforce_variations=True, old_mode=False, *args, **kwargs):
    cfg = CN()
    if old_mode:
        filename = np.random.choice(all_filenames)
        cfg.tilefile = f"{filename}"
    else:
        filename = np.random.choice(all_filenames)
        file_ind = np.random.choice(filename_to_indices[filename])
        cfg.tilefile = f"{filename}_{file_ind}.png"
    if enforce_variations:
        cfg.tile_effects = sample_tile_effects()
    else:
        if np.random.choice([True, False]):
            cfg.tile_effects = get_default_tile_effects()
        else:
            cfg.tile_effects = sample_tile_effects()
    return cfg

@add_to_registry(tile_variable_mapper, "tile")
def var_resampler_tile(current_cfg, all_filenames, filename_to_indices, *args, **kwargs):
    resamplers = {}
    def replace_tilefile():
        filename = np.random.choice(all_filenames)
        file_ind = np.random.choice(filename_to_indices[filename])
        def edit_func(cfg):
            cfg.tilefile = f"{filename}_{file_ind}.png"
            return cfg
        return edit_func
    def replace_tile_effects():
        tile_effects = sample_tile_effects()
        def edit_func(cfg):
            cfg.tile_effects = tile_effects
            return cfg
        return edit_func
    resamplers["tilefile"] = replace_tilefile
    resamplers["tile_effects"] = replace_tile_effects
    return resamplers

@add_to_registry(tile_default_mapper, "tile")
def get_default_tile(*args, **kwargs):
    cfg = CN()
    index = kwargs.get("index", 0)
    cfg.tilefile = BASE_TILES[index]
    cfg.tile_effects = get_default_tile_effects()
    return cfg

######## TILE Effects #######

@add_to_registry(tile_sampler_mapper, "tile_effects")
def sample_tile_effects(*args, **kwargs):
    cfg = CN()
    if np.random.uniform(0, 1) < 0.1:
        do_rotate = True
        rot = np.random.uniform(-np.pi/4, np.pi/4)
    else:
        do_rotate = False
        rot = 0
    if np.random.uniform(0, 1) < 0.1:
        do_scale = True
        scale = np.random.uniform(0.5, 1)
    else:
        do_scale = False
        scale = 1
    if np.random.uniform(0, 1) < 0.1:
        ref_x = True
    else:
        ref_x = False
    if np.random.uniform(0, 1) < 0.1:
        ref_y = True
    else:
        ref_y = False
    if np.random.uniform(0, 1) < 0.1:
        do_recolor = True
        new_color_hue = np.random.uniform(0, 1)
        # This should be decided at test time with all the information.
    else:
        do_recolor = False
        new_color_hue = 0
    if np.random.uniform(0, 1) < 0.1:
        # Do shadow
        do_shadow = True
        shadow_thickness = np.random.uniform(SHADOW_MIN, SHADOW_MAX)
    else:
        do_shadow = False
        shadow_thickness = 0
    if not do_shadow and np.random.uniform(0, 1) < 0.1:
        do_outline = True
        outline_thickness = np.random.uniform(OUTLINE_MIN, OUTLINE_MAX)
        if np.random.choice([True, False]):
            outline_color = (0, 0, 0)
        else:
            outline_color = np.random.uniform(0, 1, 3)
            # add alpha
        outline_color = (*outline_color, 1)
    else:
        do_outline = False
        outline_thickness = 0
        outline_color = (0, 0, 0)
    
    if np.random.uniform(0, 1) < 0.1:
        do_opacity = True
        opacity = np.random.uniform(0.5, 1)
    else:
        do_opacity = False
        opacity = 1
    cfg.do_rotate = bool(do_rotate)
    cfg.rot = float(rot)
    cfg.do_scale = bool(do_scale)
    cfg.scale = float(scale)
    cfg.ref_x = bool(ref_x)
    cfg.ref_y = bool(ref_y)
    cfg.do_recolor = bool(do_recolor)
    cfg.new_color_hue = int(new_color_hue)
    cfg.do_outline = bool(do_outline)
    cfg.outline_thickness = float(outline_thickness)
    cfg.outline_color = outline_color
    cfg.do_shadow = bool(do_shadow)
    cfg.shadow_thickness = float(shadow_thickness)
    cfg.do_opacity = bool(do_opacity)
    cfg.opacity = float(opacity)
    return cfg

@add_to_registry(tile_variable_mapper, "tile_effects")
def var_resampler_tile_effects(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_do_rotate():
        do_rotate = np.random.choice([True, False])
        rot = np.random.uniform(-np.pi/4, np.pi/4)
        def edit_func(cfg):
            cfg.do_rotate = do_rotate
            cfg.rot = rot
            return cfg
        return edit_func
    def replace_do_scale():
        do_scale = np.random.choice([True, False])
        scale = np.random.uniform(0.5, 1)
        def edit_func(cfg):
            cfg.do_scale = do_scale
            cfg.scale = scale
            return cfg
        return edit_func
    def replace_do_ref_x():
        ref_x = np.random.choice([True, False])
        def edit_func(cfg):
            cfg.ref_x = ref_x
            return cfg
        return edit_func
    def replace_do_ref_y():
        ref_y = np.random.choice([True, False])
        def edit_func(cfg):
            cfg.ref_y = ref_y
            return cfg
        return edit_func
    def replace_do_recolor():
        do_recolor = np.random.choice([True, False])
        # recolor_seed = np.random.randint(0, 100000)
        new_color_hue = np.random.uniform(0, 1)
        def edit_func(cfg):
            cfg.do_recolor = do_recolor
            cfg.new_color_hue = new_color_hue
            return cfg
        return edit_func
    def replace_do_outline():
        do_outline = np.random.choice([True, False])
        outline_thickness = np.random.uniform(OUTLINE_MIN, OUTLINE_MAX)
        if np.random.choice([True, False]):
            outline_color = (0, 0, 0)
        else:
            outline_color = np.random.uniform(0, 1, 3)
        def edit_func(cfg):
            cfg.do_outline = do_outline
            cfg.outline_thickness = outline_thickness
            cfg.outline_color = outline_color
            return cfg
        return edit_func
    def replace_do_shadow():
        do_shadow = np.random.choice([True, False])
        shadow_thickness = np.random.uniform(SHADOW_MIN, SHADOW_MAX)
        def edit_func(cfg):
            cfg.do_shadow = do_shadow
            cfg.shadow_thickness = shadow_thickness
            return cfg
        return edit_func
    def replace_do_opacity():
        do_opacity = np.random.choice([True, False])
        opacity = np.random.uniform(0.5, 1)
        def edit_func(cfg):
            cfg.do_opacity = do_opacity
            cfg.opacity = opacity
            return cfg
        return edit_func
    resamplers["do_rotate"] = replace_do_rotate
    resamplers["rot"] = None
    resamplers["do_scale"] = replace_do_scale
    resamplers["scale"] = None
    resamplers["ref_x"] = replace_do_ref_x
    resamplers["ref_y"] = replace_do_ref_y
    resamplers["do_recolor"] = replace_do_recolor
    resamplers["new_color_hue"] = None
    resamplers["do_outline"] = replace_do_outline
    resamplers["outline_thickness"] = None
    resamplers["outline_color"] = None
    resamplers["do_shadow"] = replace_do_shadow
    resamplers["shadow_thickness"] = None
    resamplers["do_opacity"] = replace_do_opacity
    resamplers["opacity"] = None
    return resamplers

@add_to_registry(tile_default_mapper, "tile_effects")
def get_default_tile_effects(*args, **kwargs):
    cfg = CN()
    cfg.do_rotate = False
    cfg.rot = 0
    cfg.do_scale = False
    cfg.scale = 1
    cfg.ref_x = False
    cfg.ref_y = False
    cfg.do_recolor = False
    cfg.new_color_hue = 0
    cfg.do_outline = False
    cfg.outline_thickness = 0
    cfg.outline_color = (0, 0, 0)
    cfg.do_shadow = False
    cfg.shadow_thickness = 0
    cfg.do_opacity = False
    cfg.opacity = 1
    return cfg

N_TILES = 4
N_TILE_EFFECTS = 4

@add_to_registry(tile_sampler_mapper, "fill_tileset")
def fill_sample_tileset(tileset, n_tiles=None, *args, **kwargs):
    cfg = CN()
    # variations: 
    tile_one = sample_param_tile(tileset, *args, **kwargs)
    tile_two = sample_param_tile(tileset, *args, **kwargs)
    tile_three = sample_param_tile(tileset, *args, **kwargs)
    tile_four = sample_param_tile(tileset, *args, **kwargs)
    # Get a sequence of four names, and replace them.
    tileset = [tile_one, tile_two, tile_three, tile_four]

    if n_tiles is None:
        n_tiles = int(np.random.choice([1, 2, 3, 4], p=[0.66, 0.2, 0.1, 0.04]))
    else:
        n_tiles = int(n_tiles)
    
    tile_order = list(np.random.choice([0, 1, 2, 3], n_tiles, replace=False))
    sel_tiles = [tileset[x] for x in tile_order]
    cfg.tileset = sel_tiles
    return cfg

@add_to_registry(tile_sampler_mapper, "param_tile")
def sample_param_tile(enforce_variations=True, *args, **kwargs):
    cfg = CN()
    cfg.tile_ind = int(np.random.choice(range(N_TILES)))
    cfg.effect_ind = np.random.choice(range(N_TILE_EFFECTS))
    if enforce_variations:
        cfg.tile_effects = get_default_tile_effects()
    else:
        if np.random.choice([True, False]):
            cfg.tile_effects = get_default_tile_effects()
        else:
            cfg.tile_effects = sample_tile_effects()
    return cfg

from yacs.config import CfgNode as CN
import numpy as np
# add a layout specs -> get the init layout and 
from .mtp_layout import fill_sample_layout, random_fill_sample_layout
from .mtp_cellfx import fill_sample_mtp_cellfx
from .mtp_tile import sample_tileset

# Need to update the tileset -> Only simple primitives. 

def sample_fill(*args, **kwargs):
    cfg = CN()
    cfg._type = "Fill"
    if "n_tiles" in kwargs:
        kwargs['n_tiles'] = 1
    cfg.tileset = sample_tileset(*args, **kwargs)
    n_tiles = len(cfg.tileset.tileset)
    cfg.layout = fill_sample_layout(n_tiles=n_tiles)
    cfg.cellfx = fill_sample_mtp_cellfx(n_tiles=n_tiles)
    # FIll specific
    cfg.stochastic_drop = False # bool(np.random.choice([True, False]))
    cfg.drop_rate = np.random.uniform(0.1, 0.5)
    cfg.fill_mode = 0
    cfg.dilate = True # bool(np.random.choice([True, False]))
    cfg.dilate_amount = np.random.uniform(0.005, 0.002)
    return cfg


rect_repeat_split_types = ["RectRepeat", "HexRepeat", "HexRepeatY", 
                "RectRepeatShiftedX", "RectRepeatShiftedY"]
    
def sample_repeat_fill(*args, **kwargs):
    cfg = sample_fill(*args, **kwargs)
    cfg._type = "RepeatFill"
    fill_cutoff = np.random.uniform(0.0, 0.2)
    cfg.fill_cutoff = float(fill_cutoff)
    seed = np.random.randint(0, 100000)
    cfg.seed = int(seed)
    return cfg

## For the Tile level version.

def random_fill_rect_repeat(tileset, cellfx, split_type="RandomFillRectRepeat", *args, **kwargs):
    cfg = CN()
    cfg._type = "RandomFillRectRepeat"
    cfg.tileset = tileset
    n_tiles = len(cfg.tileset.tileset)
    if split_type == "RandomFillRectRepeat":
        cfg.layout = random_fill_sample_layout(split_types = rect_repeat_split_types)
    else:
        cfg.layout = random_fill_sample_layout(split_types = ["VoronoiRepeat"])
    cfg.cellfx = cellfx
    # FIll specific
    cfg.stochastic_drop = True
    cfg.drop_rate = np.random.uniform(0.1, 0.2)
    cfg.fill_mode = 0
    cfg.dilate = True 
    cfg.dilate_amount = np.random.uniform(0.005, 0.002)
    fill_cutoff = np.random.uniform(0.0, 0.2)
    cfg.fill_cutoff = float(fill_cutoff)
    seed = np.random.randint(0, 100000)
    cfg.seed = int(seed)
    return cfg
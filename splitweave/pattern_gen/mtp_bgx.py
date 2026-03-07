# border config, 
# bacground config
# Fill config.
from yacs.config import CfgNode as CN
import numpy as np
from .signal_2d_func import discrete_signal_specs
from .mtp_layout import bg_sample_layout
from .mtp_cellfx import sample_recolor_fx
bg_modes = [
    "plain",
    "grid_n_color",
]

border_modes = ["simple", "dotted", "onion", "siny"]


def sample_bg():
    bg_specs = CN()
    bg_specs._type = "Background"

    bg_mode = str(np.random.choice(bg_modes, p=[0.65, 0.35]))
    seed = np.random.randint(0, 1000000)
    if not bg_mode == "plain":
        grid = str(np.random.choice(["original", "new"]))
        bg_specs.color_fx = sample_recolor_fx(None)
        bg_specs.grid = grid
        if grid == "new":
            layout_config = bg_sample_layout(n_tiles=1, simple_types=True)
            bg_specs.layout_config = layout_config
    bg_specs.bg_mode = bg_mode 
    bg_specs.color_seed = int(seed)

    
    return bg_specs

def sample_default_bg():
    bg_specs = CN()
    bg_specs._type = "Background"

    bg_mode = "plain"
    seed = np.random.randint(0, 1000000)
    bg_specs.bg_mode = bg_mode 
    bg_specs.color_seed = int(seed)
    return bg_specs

def sample_border(layout_config):
    border_spec = CN()
    border_spec._type = "Border"
    border_mode = str(np.random.choice(border_modes))
    border_spec.border_mode = border_mode
    border_size = np.random.uniform(0.005, 0.01)
    if border_mode == "siny":
        border_size = border_size / 5
    # color with seed or other?
    color_seed = np.random.randint(0, 1000000)
    if np.random.choice([True, False]):
        signal = discrete_signal_specs(allow_noisy_central_pivot=True, avoid_random=True)
        border_spec.signal = signal
    if hasattr(layout_config.split, "seed"):
        border_spec.exec_seed = layout_config.split.seed

    border_spec.mode = "single"
    border_spec.border_size = float(border_size)
    border_spec.color_seed = int(color_seed)
    return border_spec

def default_border(layout_config):

    border_spec = CN()
    border_spec._type = "Border"
    border_mode = "simple"
    border_spec.border_mode = border_mode
    border_size = 0.01
    # color with seed or other?
    color_seed = np.random.randint(0, 1000000)
    if hasattr(layout_config.split, "seed"):
        border_spec.exec_seed = layout_config.split.seed

    border_spec.mode = "single"
    border_spec.border_size = float(border_size)
    border_spec.color_seed = int(color_seed)
    return border_spec

def sample_fill():
    fill_specs = CN()
    fill_specs._type = "Fill"
    fill_mode = np.random.choice([0, 1, 2, 3])
    fill_specs.fill_mode = int(fill_mode)
    if fill_mode == 1:
        fill_specs.color_seed = np.random.randint(0, 1000000)
        signal = discrete_signal_specs(allow_noisy_central_pivot=False, avoid_random=True)
        fill_specs.signal = signal
    return fill_specs


def other_configs():
    # if BG is plain then can have a fill.
    bg_specs = sample_bg()
    if np.random.choice([True, False]):
        border_specs = sample_border(bg_specs.layout_config)
        fill_specs = None
    else:
        border_specs = None
        if bg_specs.bg_mode in ["plain"]:
            fill_specs = sample_fill()
        else:
            fill_specs = None
    return bg_specs, border_specs, fill_specs
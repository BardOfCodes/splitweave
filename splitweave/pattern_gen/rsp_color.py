import torch as th
import numpy as np
from yacs.config import CfgNode as CN
from .colors import tetradic_generator
from .mtp_layout import bg_sample_layout
# k-random color fill.

# k-random color fill.
fill_modes = ['solid', 'lines', 'squiggle', "dots"]
color_modes = ["simple", "discrete_interp", "discrete_tri_interp"]
color_modes_l = [0.7, 0.15, 0.15]
def sample_fill_cfg(n_colors=None, *args, **kwargs):
    fill_specs = CN()
    fill_mode = np.random.choice(fill_modes)
    fill_mode = "solid"
    fill_specs.fill_mode = str(fill_mode)
    do_modulate = False
    if fill_mode == "solid":
        coloring_mode = np.random.choice(color_modes, p=color_modes_l)
        fill_specs.coloring_mode = str(coloring_mode)
        if coloring_mode == "simple":
            if n_colors is None:
                n_colors = np.random.choice([2, 3, 4, 5, 6])
                n_colors = int(n_colors)
            palette = tetradic_generator(n_colors)
            palette = [np.array(x.to_rgb()) for x in palette]
            if n_colors == 4:
                temp_mode = np.random.choice(['simple', 'mix', 'alternate'])
                if temp_mode == 'mix':
                    palette[1] = (palette[0] + palette[1]) / 2
                    palette[3] = palette[1]
                elif temp_mode == 'alternate':
                    palette[1] = palette[3]
            if n_colors == 6:
                temp_mode = np.random.choice(['simple', 'mix', 'alternate'])
                if temp_mode == 'mix':
                    palette[1] = (palette[0] * 0.75 + palette[3] * 0.25) / 2
                    palette[2] = (palette[0] * 0.25 + palette[3] * 0.75) / 2
                    palette[4] = palette[2]
                    palette[5] = palette[1]
                elif temp_mode == 'alternate':
                    palette[1] = palette[3]
                    palette[5] = palette[3]
            #  Now sometimes we can do the mix,
            if False:
            # if np.random.choice([True, False]):
                # do modulator.
                # just one effect. 
                do_modulate = True
                fill_modulator = ["simple" for _ in range(n_colors)]
                new_effect = np.random.choice(["stripes", "squiggle", "dots", "checkerboard"])
                new_effect = str(new_effect)
                # Sample parameters according to the effect. 
                if new_effect == "stripes": 
                    modulate_param = sample_stripes()
                elif new_effect == "squiggle":
                    modulate_param = sample_squiggly()
                elif new_effect == "dots":
                    modulate_param = sample_dotted()
                elif new_effect == "checkerboard":
                    modulate_param = sample_checkerboard()
                if n_colors == 4:
                    if temp_mode == 'mix':
                        fill_modulator[1] = new_effect
                        fill_modulator[3] = new_effect
                    elif temp_mode == 'alternate':
                        fill_modulator[1] = new_effect
                        fill_modulator[3] = new_effect
                elif n_colors == 6:
                    if temp_mode == 'mix':
                        fill_modulator[1] = new_effect
                        fill_modulator[3] = new_effect
                        fill_modulator[5] = new_effect
                    elif temp_mode == 'alternate':
                        fill_modulator[1] = new_effect
                        fill_modulator[3] = new_effect
                        fill_modulator[5] = new_effect
                else:
                    for i in range(n_colors):
                        if np.random.choice([True, False]):
                            fill_modulator[i] = new_effect
                        if all([x != "simple" for x in fill_modulator]):
                            fill_modulator[0] = "simple"
            else:
                do_modulate = False
                fill_modulator = ["simple" for _ in range(n_colors)]
        elif coloring_mode == "discrete_interp":
            n_colors = 2# np.random.choice([2, 3])
            palette = tetradic_generator(n_colors)
            if np.random.choice([True, False]):
                color_lightness_1 = np.random.uniform(0.7, 1.0)
                color_lightness_2 = np.random.uniform(0.0, 0.3)
            else:
                color_lightness_1 = np.random.uniform(0.0, 0.3)
                color_lightness_2 = np.random.uniform(0.7, 1.0)
            palette[0].lightness = float(color_lightness_1)
            palette[1].lightness = float(color_lightness_2)
            palette = [np.array(x.to_rgb()) for x in palette]
            interp_mode = np.random.choice(["simple", "symmetric", "siny"])
            fill_specs.interp_mode = str(interp_mode)
            fill_specs.sin_k = int(np.random.randint(1, 3))
            fill_specs.mid_point = np.random.uniform(0.3, 0.7)
        elif coloring_mode == "discrete_tri_interp":
            n_colors = 3# np.random.choice([2, 3])
            palette = tetradic_generator(n_colors)
            if np.random.choice([True, False]):
                color_lightness_1 = np.random.uniform(0.7, 1.0)
                color_lightness_3 = np.random.uniform(0.0, 0.3)
            else:
                color_lightness_1 = np.random.uniform(0.0, 0.3)
                color_lightness_3 = np.random.uniform(0.7, 1.0)

            palette[0].lightness = float(color_lightness_1)
            palette[1].lightness = 0.5
            palette[2].lightness = float(color_lightness_3)
            palette = [np.array(x.to_rgb()) for x in palette]
            fill_specs.mid_point = np.random.uniform(0.3, 0.7)
            interp_mode = np.random.choice(["simple", "symmetric", "siny"])
            fill_specs.interp_mode = str(interp_mode)
            fill_specs.sin_k = int(np.random.randint(1, 3))
                                     
        fill_specs.color_palette = palette
        fill_specs.n_colors = n_colors
    # would require borders etc. -> I think not worth it, we can specify it later in the future. 
    if do_modulate:
        fill_specs.fill_modulator = fill_modulator
        fill_specs.fill_modulator_params = modulate_param
        fill_bg_palette = tetradic_generator(1)
        fill_bg_palette = [np.array(x.to_rgb()) for x in fill_bg_palette]
        fill_specs.modulate_bg_color = fill_bg_palette[0:1]
        fill_specs.layout_specs = bg_sample_layout(n_tiles=1, simple_types=True)
    fill_specs.do_modulate = do_modulate

    return fill_specs



def sample_squiggly():
    freq = np.random.choice([10, 5, 15])
    shift_amp = np.random.choice([0.01, 0.02, 0.03])
    freq_2 = np.random.choice([1, 2, 3])
    dir = np.random.choice(['x', 'y', 'diag', 'flip_diag', 'radial', 'dist'])
    options = [(0, 0), (1, 0), (0, 1), (1, 1), 
                (-1, 0), (-1, -1), (0, -1), (1, -1), (-1, 1)]
    param = options[np.random.choice(range(len(options)))]
    origin_shift_x = param[0]
    origin_shift_y = param[1]
    erode_amount = 0.7
    sig_rate = 0.1
    params = [freq, shift_amp, freq_2, dir, origin_shift_x, origin_shift_y, erode_amount, sig_rate]
    return params


def sample_dotted():
    freq = np.random.randint(5, 15)
    dot_size = np.random.uniform(0.001, 0.002)
    offset = np.random.uniform(0.0, 1.0)
    sig_rate = 0.1
    params = [freq, dot_size, offset, sig_rate]
    return params

def sample_checkerboard():
    freq = np.random.randint(5, 15)
    dot_size = np.random.uniform(0.001, 0.002)
    offset = np.random.uniform(0.0, 1.0)
    sig_rate = 0.1
    params = [freq, dot_size, offset, sig_rate]
    return params

def sample_stripes():
    freq = np.random.choice([10, 5, 15])
    width = np.random.choice([0.01, 0.02, 0.03])
    dir = np.random.choice(['x', 'y', 'diag', 'flip_diag', 'radial'])
    options = [(0, 0), (1, 0), (0, 1), (1, 1), 
                (-1, 0), (-1, -1), (0, -1), (1, -1), (-1, 1)]
    param = options[np.random.choice(range(len(options)))]
    shift_x = param[0]
    shift_y = param[1]
    erode_amount = 0.7

    sig_rate = 0.1
    params = [freq, width, dir, shift_x, shift_y, erode_amount, sig_rate]
    return params
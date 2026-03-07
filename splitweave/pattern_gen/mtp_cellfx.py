
# Will have a SIGNAL 
# CELLFX -> SIGNAL, VALUES
import scipy.stats as spyst
import numpy as np
from yacs.config import CfgNode as CN
from .utils import add_to_registry
from .signal_2d_func import sample_signal, default_signal, continuous_signal_specs, discrete_signal_specs
REFLECTION_MODES = ["x", "y", "xy"]

cellfx_sampler_mapper = {}
cellfx_variable_mapper = {}
cellfx_default_mapper = {}

# FXs
# Translate
# Rotate
# Scale
# Reflect
# Tile Ordering
# Recolor
# Outline
# Shadow
# Opacity


###### Tile Ordering ######
def sample_tile_ordering_fx(parent_cfg, n_tiles=None, *args, **kwargs):
    cfg = CN()
    cfg._type = "TileOrderFx"
    # Needs a discrete signal...
    # contains its own signal
    if n_tiles is None:
        raise ValueError("n_tiles should be provided")
        n_tiles = int(np.random.choice([1, 2, 3, 4], p=[0.66, 0.2, 0.1, 0.04]))
    else:
        n_tiles = int(n_tiles)
    if n_tiles > 1:
        # randomly assign 0, 1, 2, 3 to n tiles without replacement
        cfg.signal = discrete_signal_specs(k=n_tiles, allow_noisy_central_pivot=False, no_count=True,
                                           avoid_random=True)
        if cfg.signal.discrete_mode in ["xx", "yy", "diag_1", "diag_2"]:
            cfg.signal.k = n_tiles - 1 
    cfg.n_tiles = int(n_tiles)
    return cfg

def var_resampler_tile_ordering_fx(current_cfg, *args, **kwargs):
    resamplers = {}
    old_n_tiles = current_cfg.n_tiles
    if old_n_tiles == 1:
        # Additive change -> Not allowed
        resamplers["n_tiles"] = None
    else:
        def replace_n_tiles():
            n_tiles = np.random.randint(1, old_n_tiles + 1)
            def edit_func(cfg):
                cfg.n_tiles = int(n_tiles)
                return cfg
            return edit_func
        resamplers["n_tiles"] = replace_n_tiles
    # Never change the tile order?
    def replace_tile_order():
        n_tiles = current_cfg.n_tiles
        cur_tile_order = current_cfg.tile_order
        # shuffle the current tile order
        new_tile_order = list(np.random.choice(cur_tile_order, n_tiles, replace=False))
        def edit_func(cfg):
            if cfg.n_tiles != n_tiles:
                cfg.n_tiles = n_tiles
            cfg.tile_order = new_tile_order
            return cfg
        return edit_func
    
    def replace_signal():
        def edit_func(cfg):
            cfg.signal = discrete_signal_specs(allow_noisy_central_pivot=False)
            cfg.signal.k = cfg.n_tiles
            return cfg
        return edit_func
    resamplers["tile_order"] = replace_tile_order
    resamplers["signal"] = replace_signal
    return resamplers

def default_tile_ordering_fx():
    cfg = CN()
    cfg._type = "TileOrderFx"
    cfg.n_tiles = 1

    return cfg

######## Translate FX ########
@add_to_registry(cellfx_sampler_mapper, "TranslateFx")
def sample_translate_fx(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "TranslateFx"

    translation_x = np.random.uniform(.2, .4)
    translation_x = np.random.choice([1, -1]) * translation_x
    translation_y = np.random.uniform(.2, .4)
    translation_y = np.random.choice([1, -1]) * translation_y
    mode = np.random.choice(["single", "double"])
    cfg.t_x = float(translation_x)
    cfg.t_y = float(translation_y)
    cfg.mode = str(mode)
    cfg.signal = discrete_signal_specs(allow_noisy_central_pivot=False, avoid_random=True)

    return cfg

@add_to_registry(cellfx_variable_mapper, "TranslateFx")
def var_resampler_translate_fx(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_t_x_t_y():
        translation_x = np.random.uniform(.2, .4)
        translation_x = np.random.choice([1, -1]) * translation_x
        translation_y = np.random.uniform(.2, .4)
        translation_y = np.random.choice([1, -1]) * translation_y
        def edit_func(cfg):
            cfg.t_x = translation_x
            cfg.t_y = translation_y
            return cfg
        return edit_func
    resamplers["t_x"] = replace_t_x_t_y
    resamplers["t_y"] = replace_t_x_t_y
    resamplers["mode"] = None
    # The signal confs?

    return resamplers

@add_to_registry(cellfx_default_mapper, "TranslateFx")
def default_translate_fx():
    cfg = CN()
    cfg._type = "TranslateFx"
    cfg.t_x = 0.1
    cfg.t_y = 0.1
    cfg.mode = "single"
    return cfg

######## Rotate FX ########

@add_to_registry(cellfx_sampler_mapper, "RotateFx")
def sample_rotate_fx(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "RotateFx"
    rotation = np.random.uniform(np.pi/4, np.pi/2)
    rotation = np.random.choice([1, -1]) * rotation
    cfg.rotation = float(rotation)
    mode = np.random.choice(["single", "double"])
    cfg.mode = str(mode)
    cfg.signal = discrete_signal_specs(allow_noisy_central_pivot=False, avoid_random=True)
    if cfg.signal.k == 2:
        cfg.rotation = cfg.rotation / 2
    return cfg

@add_to_registry(cellfx_variable_mapper, "RotateFx")
def var_resampler_rotate_fx(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_rotation():
        rotation = np.random.uniform(np.pi/6, np.pi/2)
        rotation = np.random.choice([1, -1]) * rotation
        def edit_func(cfg):
            cfg.rotation = rotation
            return cfg
        return edit_func
    resamplers["rotation"] = replace_rotation
    resamplers["mode"] = None
    return resamplers

@add_to_registry(cellfx_default_mapper, "RotateFx")
def default_rotate_fx():
    cfg = CN()
    cfg._type = "RotateFx"
    cfg.rotation = np.pi/4
    cfg.mode = "single"
    return cfg

# Scale FX
@add_to_registry(cellfx_sampler_mapper, "ScaleFx")
def sample_scale_fx(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "ScaleFx"
    scale = np.random.uniform(0.25, 0.75)
    cfg.scale = float(scale)
    cfg.mode = "single"
    cfg.signal = discrete_signal_specs(allow_noisy_central_pivot=False, avoid_random=True)
    if cfg.signal.k == 2:
        cfg.scale = cfg.scale - 0.25
    return cfg

@add_to_registry(cellfx_variable_mapper, "ScaleFx")
def var_resampler_scale_fx(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_scale():
        scale = np.random.uniform(0.25, .75)
        def edit_func(cfg):
            cfg.scale = scale
            return cfg
        return edit_func
    resamplers["scale"] = replace_scale
    resamplers["mode"] = None
    return resamplers

@add_to_registry(cellfx_default_mapper, "ScaleFx")
def default_scale_fx():
    cfg = CN()
    cfg._type = "ScaleFx"
    cfg.scale = 0.5
    cfg.mode = "single"
    return cfg

######## Reflect FX ########

@add_to_registry(cellfx_sampler_mapper, "ReflectFx")
def sample_reflect_fx(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "ReflectFx"
    reflection = np.random.choice(REFLECTION_MODES)
    cfg.reflect = str(reflection)
    cfg.mode = "single"
    cfg.signal = discrete_signal_specs(k=2, allow_noisy_central_pivot=False, avoid_random=True)
    return cfg

@add_to_registry(cellfx_variable_mapper, "ReflectFx")
def var_resampler_reflect_fx(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_reflect():
        reflection = np.random.choice(REFLECTION_MODES)
        def edit_func(cfg):
            cfg.reflect = reflection
            return cfg
        return edit_func
    resamplers["reflect"] = replace_reflect
    return resamplers

@add_to_registry(cellfx_default_mapper, "ReflectFx")
def default_reflect_fx():
    cfg = CN()
    cfg._type = "ReflectFx"
    cfg.mode = "single"
    cfg.reflect = "x"
    return cfg

######## Recolor FX ########

@add_to_registry(cellfx_sampler_mapper, "RecolorFx")
def sample_recolor_fx(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "RecolorFx"
    # smooth or discrete
    cfg.recolor_type = str(np.random.choice(["smooth", "discrete"]))
    cfg.recolor_seed = np.random.randint(0, 100000)
    cfg.mode = str(np.random.choice(["single", "double"]))
    cfg.signal = discrete_signal_specs(allow_noisy_central_pivot=False)
    return cfg

@add_to_registry(cellfx_variable_mapper, "RecolorFx")
def var_resampler_recolor_fx(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_recolor_seed():
        recolor_seed = np.random.randint(0, 100000)
        def edit_func(cfg):
            cfg.recolor_seed = recolor_seed
            return cfg
        return edit_func
    resamplers["recolor_seed"] = replace_recolor_seed
    return resamplers

@add_to_registry(cellfx_default_mapper, "RecolorFx")
def default_recolor_fx():
    cfg = CN()
    cfg._type = "RecolorFx"
    cfg.recolor_seed = 0
    return cfg

# outline fx
@add_to_registry(cellfx_sampler_mapper, "OutlineFx")
def sample_outline_fx(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "OutlineFx"
    cfg.outline_color = tuple(list(np.random.uniform(0, 1, 3)) + [1.0])
    cfg.outline_thickness = float(np.random.uniform(0.25, 0.5))
    cfg.mode = "single"
    cfg.signal = discrete_signal_specs(allow_noisy_central_pivot=False, avoid_random=True)
    return cfg

@add_to_registry(cellfx_variable_mapper, "OutlineFx")
def var_resampler_outline_fx(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_outline_seed():
        outline_seed = np.random.randint(0, 100000)
        def edit_func(cfg):
            cfg.outline_seed = outline_seed
            return cfg
        return edit_func
    def replace_outline_thickness():
        outline_thickness = np.random.uniform(0.01, 0.1)
        def edit_func(cfg):
            cfg.outline_thickness = outline_thickness
            return cfg
        return edit_func
    resamplers["outline_seed"] = replace_outline_seed
    resamplers["outline_thickness"] = replace_outline_thickness
    return resamplers

@add_to_registry(cellfx_default_mapper, "OutlineFx")
def default_outline_fx():
    cfg = CN()
    cfg._type = "OutlineFx"
    cfg.outline_seed = 0
    cfg.outline_thickness = 0.05
    return cfg

# shadow fx
# @add_to_registry(cellfx_sampler_mapper, "ShadowFx")
def sample_shadow_fx(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "ShadowFx"
    cfg.shadow_thickness = float(np.random.uniform(0.25, 0.35))
    shift_x = np.random.uniform(0.1, 0.2)
    shift_x = np.random.choice([1, -1]) * shift_x
    shift_y = np.random.uniform(0.1, 0.2)
    shift_y = np.random.choice([1, -1]) * shift_y
    cfg.shift_x = float(shift_x)
    cfg.shift_y = float(shift_y)
    cfg.mode = "single"
    cfg.signal = discrete_signal_specs(allow_noisy_central_pivot=False)
    return cfg

# @add_to_registry(cellfx_variable_mapper, "ShadowFx")
def var_resampler_shadow_fx(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_shadow_offset():
        shadow_offset = np.random.uniform(0.01, 0.1)
        def edit_func(cfg):
            cfg.shadow_offset = shadow_offset
            return cfg
        return edit_func
    resamplers["shadow_offset"] = replace_shadow_offset
    return resamplers

# @add_to_registry(cellfx_default_mapper, "ShadowFx")
def default_shadow_fx():
    cfg = CN()
    cfg._type = "ShadowFx"
    cfg.shadow_offset = 0.05
    return cfg

### Apply alpha effect: 

@add_to_registry(cellfx_sampler_mapper, "AlphaModFX")
def sample_outline_fx(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "AlphaModFX"
    
    cfg.outline_color = tuple(list(np.random.uniform(0, 1, 3)) + [1.0])
    cfg.outline_thickness = float(np.random.uniform(0.25, 0.5))
    cfg.mode = "single"
    cfg.signal = discrete_signal_specs(allow_noisy_central_pivot=False, avoid_random=True)
    return cfg

@add_to_registry(cellfx_variable_mapper, "AlphaModFX")
def var_resampler_outline_fx(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_outline_seed():
        outline_seed = np.random.randint(0, 100000)
        def edit_func(cfg):
            cfg.outline_seed = outline_seed
            return cfg
        return edit_func
    def replace_outline_thickness():
        outline_thickness = np.random.uniform(0.01, 0.1)
        def edit_func(cfg):
            cfg.outline_thickness = outline_thickness
            return cfg
        return edit_func
    resamplers["outline_seed"] = replace_outline_seed
    resamplers["outline_thickness"] = replace_outline_thickness
    return resamplers

@add_to_registry(cellfx_default_mapper, "AlphaModFX")
def default_outline_fx():
    cfg = CN()
    cfg._type = "AlphaModFX"
    cfg.outline_seed = 0
    cfg.outline_thickness = 0.05
    return cfg


# Opacity
@add_to_registry(cellfx_sampler_mapper, "OpacityFx")
def sample_opacity_fx(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "OpacityFx"
    cfg.signal = discrete_signal_specs(allow_noisy_central_pivot=False, avoid_random=True)
    if np.random.choice([True, False]):
        cfg.opacity = 1.0
        cfg.signal.k = 2
    else:
        if np.random.choice([True, False]):
            cfg.opacity = 1.0
        else:
            cfg.opacity = np.random.uniform(0.5, 0.9)
    cfg.mode = "single"

    return cfg

@add_to_registry(cellfx_variable_mapper, "OpacityFx")
def var_resampler_opacity_fx(current_cfg, *args, **kwargs):
    resamplers = {}
    def replace_opacity():
        opacity = np.random.uniform(0.1, 1)
        def edit_func(cfg):
            cfg.opacity = opacity
            return cfg
        return edit_func
    resamplers["opacity"] = replace_opacity
    return resamplers

@add_to_registry(cellfx_default_mapper, "OpacityFx")
def default_opacity_fx():
    cfg = CN()
    cfg._type = "OpacityFx"
    cfg.opacity = 0.5
    return cfg


cellfx_mapper = {
    0: "TranslateFx",
    1: "RotateFx",
    2: "ScaleFx",
    3: "ReflectFx",
    4: "RecolorFx",
    5: "OutlineFx",
    6: "OpacityFx",
    # 6: "ShadowFx",
}



def sample_mtp_cellfx(n_tiles, partial_effects=True, allow_nil=True, avoid_t=False):
    # Create tile bank
    cfg = CN()
    cfg.tile_order = sample_tile_ordering_fx(cfg, n_tiles)

    # Create cellfx
    # use beta distribution
    valid_sizes = [0, 1, 2, 3, 4, 5]
    if not allow_nil:
        valid_sizes = valid_sizes[1:]

    min_size, max_size = min(valid_sizes), max(valid_sizes)
    beta_distr = spyst.beta(2, 3.5)
    complex_values = [beta_distr.pdf(
        (x + 1 -min_size)/(max_size-min_size + 2)) for x in valid_sizes]
    complex_values = np.array(complex_values)
    complex_values = complex_values / complex_values.sum()
    n_effects = np.random.choice(valid_sizes, p=complex_values)
    
    # select n_effects from the ceel_fx mappers without replacement
    cur_effects = {x:y for x, y in cellfx_mapper.items()}
    if partial_effects:
        del cur_effects[0]
        del cur_effects[1]
        del cur_effects[2]
    if avoid_t:
        if 0 in cur_effects:
            del cur_effects[0]

    n_effects = min(n_effects, len(cur_effects))
    effect_names = np.random.choice(list(cur_effects.keys()), n_effects, replace=False)
    effect_list = []
    signal_set = []
    for i, effect in enumerate(effect_names):
        effect_name = cellfx_mapper[effect]
        effect_cfg = cellfx_sampler_mapper[effect_name](cfg)
        effect_list.append(effect_cfg)
        # setattr(cfg.effects, effect_name, effect_cfg)
        signal_set.append(effect_cfg.signal)
    if hasattr(cfg.tile_order, "signal"):
        signal_set.append(cfg.tile_order.signal)
    if n_effects > 1:
        n_signals = np.random.choice(list(range(1, max(3, n_effects))))
        assigned_signals_inds = np.random.choice(range(n_signals), n_effects, replace=True)
        for ind in range(n_effects):
            signal_ind = assigned_signals_inds[ind]
            effect_list[ind].signal = signal_set[signal_ind]
        
        
    cfg.effects = effect_list
    # now the goal is to share the signals between the effects.
    return cfg

def fill_sample_mtp_cellfx(n_tiles):
    # Create tile bank
    cfg = CN()
    cfg.tile_order = sample_tile_ordering_fx(cfg, n_tiles)
    # Create cellfx
    n_effects = np.random.choice([0, 1, 2, 3, 4, 5])
    
    # select n_effects from the ceel_fx mappers without replacement
    reject_ids = [0, 6, 5]
    cur_effects = {x:y for x, y in cellfx_mapper.items() if x not in reject_ids}
    n_effects = min(n_effects, len(cur_effects))
    effect_names = np.random.choice(list(cur_effects.keys()), n_effects, replace=False)
    effect_list = []
    signal_set = []
    for i, effect in enumerate(effect_names):
        effect_name = cellfx_mapper[effect]
        effect_cfg = cellfx_sampler_mapper[effect_name](cfg)
        effect_list.append(effect_cfg)
        # setattr(cfg.effects, effect_name, effect_cfg)
        signal_set.append(effect_cfg.signal)
    if hasattr(cfg.tile_order, "signal"):
        signal_set.append(cfg.tile_order.signal)
    if n_effects > 1:
        n_signals = np.random.choice(list(range(1, max(3, n_effects))))
        assigned_signals_inds = np.random.choice(range(n_signals), n_effects, replace=True)
        for ind in range(n_effects):
            signal_ind = assigned_signals_inds[ind]
            effect_list[ind].signal = signal_set[signal_ind]
    cfg.effects = effect_list
    # now the goal is to share the signals between the effects.
    return cfg

def replace_mtp_cellfx(current_cfg):
    resamplers = {}
    # Jus changing signal -> NOPE. 
    # Never replace signals -> Just create effect and add. 
    # Or remove signal.
    # Allow change of certain tile related things.
    tile_ordering_sampler = var_resampler_tile_ordering_fx(current_cfg.tile_order)
    tos_renamed = {f"tile_order.{k}": v for k, v in tile_ordering_sampler.items()}
    resamplers.update(tos_renamed)

    
    # and effects
    def replace_effects():
        effects = []
        n_effects = np.random.choice([1, 2, 3, 4, 5])
        # Create cellfx
        for i in range(n_effects):
            effect = np.random.choice(list(cellfx_mapper.keys()))
            effects.append(effect)
        effects = CN()
        for i, effect in enumerate(effects):
            effect_name = cellfx_mapper[effect]
            effect_cfg = cellfx_sampler_mapper[effect_name]()
            setattr(effects, effect_name, effect_cfg)
        def edit_func(cfg):
            cfg.effects = effects.clone()
            return cfg
        return edit_func
    resamplers['effects'] = replace_effects
    # For each effect inheright the editors:
    for effect_name in current_cfg.effects:
        effect_resamples = cellfx_variable_mapper[effect_name](current_cfg.effects[effect_name])
        er_with_updated_names = {f"effects.{effect_name}.{k}": v for k, v in effect_resamples.items()}
        resamplers.update(er_with_updated_names)
    return resamplers

def default_mtp_cellfx(n_tiles=None):
    cfg = CN()
    cfg.effects = []
    if n_tiles is not None:
        cfg.tile_order = sample_tile_ordering_fx(cfg, n_tiles=n_tiles)
    else:
        cfg.tile_order = default_tile_ordering_fx()
    return cfg

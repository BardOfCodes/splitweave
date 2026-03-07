from yacs.config import CfgNode as CN
import numpy as np
import torch as th
import geolipi.symbolic as gls
from geolipi.torch_compute import recursive_evaluate, expr_to_colored_canvas
from .signal_2d_func import radial_signal_specs, perlin_signal_specs, decay_signal_specs, strip_signal_specs, swirl_signal_specs
from .signal_2d_func import AXIS_LIST
from .utils import add_to_registry
# RADIAL
R_NR_L_MIN = -0.5
R_NR_L_MAX = 1.0
R_NR_S_MIN = -0.5
R_NR_S_MIN = 0.5
# PERLIN
P_NR_MIN = 0.05
P_NR_MAX = 0.1
# DECAY
D_NR_MIN = 0.5
D_NR_MAX = 0.8
# STRIP
ST_NR_MIN = 0.02
ST_NR_MAX = 0.075
# SWIRL
SW_NR_MIN = np.pi/12
SW_NR_MAX = np.pi/4

deform_sampler_mapper = {}
deform_variable_mapper = {}
deform_default_mapper = {}

#### Radial Deformation
@add_to_registry(deform_sampler_mapper, "radial_deform")
def sample_radial_deform(parent_cfg, *args, **kwargs):
    deformation_spec = radial_signal_specs()
    deformation_spec._type = "radial_deform"
    if deformation_spec.signal_mode == "linear":
        dist_rate = np.random.uniform(R_NR_L_MIN, R_NR_L_MAX)
    else:
        dist_rate = np.random.uniform(R_NR_S_MIN, R_NR_S_MIN)
    deformation_spec.dist_rate = float(dist_rate)
    return deformation_spec

@add_to_registry(deform_variable_mapper, "radial_deform")
def var_resampler_radial_deform(current_cfg, *args, **kwargs):
    keys = ["dist_rate", "center", "signal_mode", "sin_k", "phase_shift"]
    resamplers = {x: None for x in keys}
    return resamplers

@add_to_registry(deform_default_mapper, "radial_deform")
def get_default_radial_deform(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "radial_deform"
    return cfg # should not be required. 

#### Perlin Deformation

@add_to_registry(deform_sampler_mapper, "perlin_deform")    
def perlin_deformation_specs(parent_cfg, *args, **kwargs):
    # Do Random shift along x or y.
    # It means perlin along x, y, or both
    deformation_spec = perlin_signal_specs()
    deformation_spec._type = "perlin_deform"
    deformation_spec.dist_mode = "xy" # str(np.random.choice(AXIS_LIST))
    dist_rate = np.random.choice([P_NR_MIN, P_NR_MAX])
    dist_rate = float(dist_rate)
    deformation_spec.dist_rate = dist_rate
    return deformation_spec

@add_to_registry(deform_variable_mapper, "perlin_deform")
def var_resampler_perlin_deform(current_cfg, *args, **kwargs):
    keys = ["dist_rate", "resolution_seq", "seed"]
    resamplers = {x: None for x in keys}
    return resamplers

@add_to_registry(deform_default_mapper, "perlin_deform")
def get_default_perlin_deform(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "perlin_deform"
    return cfg

#### Decay Deformation
@add_to_registry(deform_sampler_mapper, "decay_deform")
def decay_deformation_specs(parent_cfg, *args, **kwargs):
    deformation_spec = decay_signal_specs()
    deformation_spec._type = "decay_deform"
    rate = np.random.uniform(D_NR_MIN, D_NR_MAX)
    flip_switch = np.random.choice([+1, -1])
    dist_rate = flip_switch * rate
    dist_rate = float(dist_rate)
    deformation_spec.dist_rate = dist_rate
    return deformation_spec

@add_to_registry(deform_variable_mapper, "decay_deform")
def var_resampler_decay_deform(current_cfg, *args, **kwargs):
    keys = ["dist_rate", "direction", "axis"]
    resamplers = {x: None for x in keys}
    return resamplers

@add_to_registry(deform_default_mapper, "decay_deform")
def get_default_decay_deform(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "decay_deform"
    return cfg

#### Strip Deformation
@add_to_registry(deform_sampler_mapper, "strip_deform")
def strip_deformation_specs(parent_cfg, *args, **kwargs):
    # Just linear add, or sin add
    deformation_spec = strip_signal_specs()
    deformation_spec._type = "strip_deform"
    dist_rate = np.random.uniform(ST_NR_MIN, ST_NR_MAX)
    deformation_spec.dist_rate = dist_rate

    return deformation_spec

@add_to_registry(deform_variable_mapper, "strip_deform")
def var_resampler_strip_deform(current_cfg, *args, **kwargs):
    keys = ["dist_rate", "angle", "axis", "sin_k", "phase_shift"]
    resamplers = {x: None for x in keys}
    return resamplers

@add_to_registry(deform_default_mapper, "strip_deform")
def get_default_strip_deform(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "strip_deform"
    return cfg

#### Swirl Deformation
@add_to_registry(deform_sampler_mapper, "swirl_deform")
def swirl_deformation_specs(parent_cfg, *args, **kwargs):
    deformation_spec = swirl_signal_specs()
    deformation_spec._type = "swirl_deform"
    noise_rate = np.random.uniform(SW_NR_MIN, SW_NR_MAX)
    noise_rate = np.random.choice([1, -1]) * noise_rate
    noise_rate = float(noise_rate)
    deformation_spec.dist_rate = noise_rate
    return deformation_spec

@add_to_registry(deform_variable_mapper, "swirl_deform")
def var_resampler_swirl_deform(current_cfg, *args, **kwargs):
    keys = ["dist_rate", "center", "signal_mode", "sin_k", "phase_shift"]
    resamplers = {x: None for x in keys}
    return resamplers

@add_to_registry(deform_default_mapper, "swirl_deform")
def get_default_swirl_deform(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "swirl_deform"
    return cfg


#### No Deformation
@add_to_registry(deform_sampler_mapper, "no_deform")
def no_deformation_specs(parent_cfg, *args, **kwargs):
    deformation_spec = swirl_signal_specs()
    deformation_spec._type = "no_deform"
    return deformation_spec

@add_to_registry(deform_variable_mapper, "no_deform")
def var_resampler_no_deform(current_cfg, *args, **kwargs):
    keys = []
    resamplers = {x: None for x in keys}
    return resamplers

@add_to_registry(deform_default_mapper, "no_deform")
def get_default_no_deform(parent_cfg, *args, **kwargs):
    cfg = CN()
    cfg._type = "no_deform"
    return cfg


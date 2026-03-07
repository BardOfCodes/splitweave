import numpy as np
import torch as th
import geolipi.symbolic as gls
from yacs.config import CfgNode as CN
from geolipi.torch_compute import recursive_evaluate
from .utils import add_to_registry

CROP_FACTOR = 0.5
CENTER_LIST = [(0, 0), (-1, -1), (1, 1), (1, -1), (1, 1),
               (0, 1), (1, 0), (-1, 0), (0, -1),
               (0.5, 0.5), (-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5),
               (0.5, 0), (-0.5, 0), (0, 0.5), (0, -0.5)]
DIRECTIONS = [(-1, -1), (1, 1), (1, -1), (1, 1), (0, 1), (1, 0), (-1, 0), (0, -1)]
AXIS_LIST = ["x", "y", "xy"]
# Hyper parameters
CENTER_RANDOM_P = 0.2
# RADIAL
RADIAL_MODES = ["siny", "linear"]
R_SIN_K_MIN = 5
R_SIN_K_MIN = 10
# PERLIN
PERLIN_MODES = [[2, 4], [2, 4, 8], [2, 4, 8, 16]]
# DECAY
ST_SIN_K_MIN = 1
ST_SIN_K_MAX = 7
# SWIRL
SWIRL_MODES = ["siny", "linear", "sigmoid"]

# This is only required if there is going to be edits inside the signal..
signal_sampler_mapper = {}
signal_variable_mapper = {}
signal_default_mapper = {}


def base_config(crop_factor=CROP_FACTOR):
    # SET SEED as well -> before calling this function.
    config = CN()
    config.crop_x = crop_factor
    t_x = np.random.uniform(-1.0, 1.0)
    t_y = np.random.uniform(-1.0, 1.0)
    config.t_x = float(t_x)
    config.t_y = float(t_y)
    return config


def radial_signal_specs():
    signal_specs = base_config()
    signal_specs._type = "RadialSignal"
    center_random = np.random.choice([True, False], p=[CENTER_RANDOM_P, 1-CENTER_RANDOM_P])
    if center_random:
        x_center = np.random.uniform(-1, 1) 
        y_center = np.random.uniform(-1, 1)
        center = (x_center, y_center)
    else:
        center = np.random.choice(range(len(CENTER_LIST)))
        center = CENTER_LIST[center]
    signal_mode = np.random.choice(RADIAL_MODES)
    sin_k = np.random.uniform(R_SIN_K_MIN, R_SIN_K_MIN)
    phase_shift = np.random.uniform(0, 2*np.pi)
    signal_specs.center = center
    signal_specs.signal_mode = str(signal_mode)
    signal_specs.sin_k = float(sin_k)
    signal_specs.phase_shift = float(phase_shift)
    return signal_specs

def perlin_signal_specs():
    signal_specs = base_config()
    signal_specs._type = "PerlinSignal"
    resolution_seq_index = np.random.choice(range(len(PERLIN_MODES)))
    resolution_seq = PERLIN_MODES[resolution_seq_index]
    signal_specs.resolution_seq = resolution_seq
    random_seed = np.random.randint(0, 100000)
    signal_specs.seed = random_seed
    
    return signal_specs

def decay_signal_specs():
    signal_specs = base_config()
    signal_specs._type = "DecaySignal"
    direction = np.random.choice(range(len(DIRECTIONS)))
    direction = DIRECTIONS[direction]
    # increase or decrease.
    axis = np.random.choice(range(len(AXIS_LIST)))
    axis = AXIS_LIST[axis]
    axis = str(axis)
    signal_specs.direction = direction
    signal_specs.axis = axis
    return signal_specs

def strip_signal_specs():
    signal_specs = base_config()
    signal_specs._type = "StripSignal"
    angle = np.random.uniform(0, 2 * np.pi)
    angle = float(angle)
    axis = np.random.choice(range(len(AXIS_LIST)))
    axis = AXIS_LIST[axis]
    axis = str(axis)
    sin_k = np.random.uniform(ST_SIN_K_MIN, ST_SIN_K_MAX)
    phase_shift = np.random.uniform(0, 2 * np.pi)
    signal_specs.angle = angle
    signal_specs.axis = axis
    signal_specs.sin_k = float(sin_k)
    signal_specs.phase_shift = float(phase_shift)
    return signal_specs


def swirl_signal_specs():
    signal_specs = radial_signal_specs()
    signal_specs._type = "SwirlSignal"
    signal_mode = np.random.choice(SWIRL_MODES)
    signal_specs.signal_mode = str(signal_mode)
    sigmoid_rate = np.random.choice([-10, -20, -40, -60])
    sigmoid_spread_rate = np.random.uniform(0.25, 0.5)
    signal_specs.sigmoid_rate = float(sigmoid_rate)
    signal_specs.sigmoid_spread = float(sigmoid_spread_rate)
    return signal_specs

# Also the discrete signals: 
discrete_modes = [
    "checkerboard",
    "x",
    "y",
    "xx",
    "yy",
    "xy",
    "diag_1",
    "diag_2",
    "random",
    "noisy_central_pivot",
    "fully_random",
    "count_diag_1",
    "count_x",
    "count_y",
    "count_diag_2",
]
mode_1 = [
    "random",
    "noisy_central_pivot",
    "fully_random",
    "count_diag_1",
    "count_x",
    "count_y",
    "count_diag_2",
]
def discrete_signal_specs(k=None, allow_noisy_central_pivot=True, no_count=False, avoid_random=False):

    signal_specs = base_config()
    signal_specs._type = "DiscreteSignal"
    cur_discrete_modes = [x for x in discrete_modes]
    if not allow_noisy_central_pivot:
        cur_discrete_modes.remove("noisy_central_pivot")
    if no_count:
        cur_discrete_modes.remove("count_diag_1")
        cur_discrete_modes.remove("count_x")
        cur_discrete_modes.remove("count_y")
        cur_discrete_modes.remove("count_diag_2")
    if avoid_random:
        cur_discrete_modes.remove("fully_random")
        cur_discrete_modes.remove("random")


        
    signal_mode = np.random.choice(cur_discrete_modes)
    inverse = np.random.choice([True, False])
    apply_sym = np.random.choice([True, False])
    if k is None:
        k = np.random.choice([2, 3, 4, 5, 6])
    if k > 3:
        group_alternate = np.random.choice([True, False])
    else:
        group_alternate = False
    # HACKS
    if signal_mode == "xy":
        k = 2
    if signal_mode == "diag_1":
        group_alternate = False
    if signal_mode in mode_1:
        inverse = False
    double_dip = False
    signal_specs.discrete_mode = str(signal_mode)
    signal_specs.k = int(k)
    signal_specs.inverse = bool(inverse)
    signal_specs.noise_rate = float(np.random.uniform(0.4, 0.6))
    signal_specs.group_alternate = bool(group_alternate)
    signal_specs.apply_sym = bool(apply_sym)
    signal_specs.double_dip = bool(double_dip)

    if signal_mode == "noisy_central_pivot":
        signal_specs.signal = continuous_signal_specs()

    return signal_specs

continuous_signal_mapper = {
    "radial": radial_signal_specs,
    "perlin": perlin_signal_specs,
    "decay": decay_signal_specs,
    "strip": strip_signal_specs,
    "swirl": swirl_signal_specs,
}

def continuous_signal_specs():
    signal_mode = np.random.choice(list(continuous_signal_mapper.keys()))
    signal_specs = continuous_signal_mapper[signal_mode]()
    return signal_specs

def sample_signal():
    if np.random.choice([True, False]):
        return continuous_signal_specs()
    else:
        return discrete_signal_specs()


def default_signal():
    signal_specs = base_config()
    signal_specs._type = "DiscreteSignal"
    signal_mode = "checkerboard"
    signal_specs.signal_mode = str(signal_mode)
    k = 2
    signal_specs.k = int(k)
    return signal_specs


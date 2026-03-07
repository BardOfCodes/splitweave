from collections import defaultdict
import torch as th
import numpy as np
from scipy.spatial import Delaunay

from .grid_renorm import voronoi_style_normalize, get_binwise_min_max


def _unpack_vec2(param):
    """Helper to unpack a 2D size / cell_size parameter.

    Accepts:
    - Tensor of shape (2,) or (..., 2) -> uses the last dimension
    - Tuple/list of length 2
    Returns two Python floats for downstream scalar math.
    """
    if isinstance(param, th.Tensor):
        if param.ndim > 1:
            param = param.reshape(-1)
        if param.numel() != 2:
            raise ValueError(f"Expected Vec2 tensor with 2 elements, got shape {tuple(param.shape)}")
        a = float(param[0].item())
        b = float(param[1].item())
        return a, b
    elif isinstance(param, (tuple, list)) and len(param) == 2:
        return float(param[0]), float(param[1])
    else:
        raise TypeError(f"Expected Vec2-like param (Tensor/tuple/list of length 2), got {type(param)}")


_S = th.tensor([1.7320508, 1])[None, ...]
_HEX_VAL_3 = th.tensor([1, .5])[None, ...]
_TAN_PI_3 = np.tan(np.pi/3)
_SIN_PI_3 = np.sin(np.pi/3)
DIAMOND_DENOM = np.sqrt(1 + _TAN_PI_3**2)

PERLIN_VECTORS = th.tensor([[1,1], [-1,1], [1,-1], [-1,-1], [1,0], [-1,0], [0,1], [0,-1]])
CENTROIDS_UPPER_LIMIT = 1000
def dummy_func(grid, *args, **kwargs):
    return grid
    
def cart_to_polar_grid(grid, radial_unit=1, angular_unit=1):
    # convert xy to polar w.r.t. origin
    grid_x = grid[..., 0]
    grid_y = grid[..., 1]
    theta = th.atan2(grid_y, grid_x)
    r = th.sqrt(grid_x**2 + grid_y**2)
    # convert to polar units
    r = r / radial_unit
    theta =  theta / angular_unit
    new_grid = th.stack([r, theta], dim=-1)
    return new_grid

def polar_to_cart_grid(grid):
    x = grid[..., 0] * th.cos(grid[..., 1])
    y = grid[..., 0] * th.sin(grid[..., 1])
    new_grid = th.stack([x, y], dim=-1)
    return new_grid

def rect_repeat_grid(grid, size):
    width, height = _unpack_vec2(size)
    # -1 to 1 repeating
    # This should also return assignments
    max_size = min(width, height)
    new_grid, grid_ids  = rect_repeat_inner(grid, width, height, max_size)
    return new_grid, grid_ids

def rect_repeat_inner(grid, width, height, max_size):
    ratio_1 = width / max_size
    ratio_2 = height / max_size
    
    grid_x_id = grid[..., 0] // width
    grid_y_id = grid[..., 1] // height
    grid_ids = th.stack([grid_x_id, grid_y_id], dim=-1)
    new_grid_x = (grid[..., 0] % width - width / 2) * 2 / width * ratio_1
    new_grid_y = (grid[..., 1] % height - height / 2) * 2 / height * ratio_2
    new_grid = th.stack([new_grid_x, new_grid_y], dim=-1)
    return new_grid, grid_ids

def rect_repeat_grid_fitting(grid, size):
    width, height = _unpack_vec2(size)
    # -1 to 1 repeating
    # This should also return assignments
    max_size = max(width, height)
    new_grid, grid_ids = rect_repeat_inner(grid, width, height, max_size)
    return new_grid, grid_ids

def rect_repeat_edge_grid(grid, size):
    width, height = _unpack_vec2(size)
    # Calculate the distance from each edge, and select min
    grid_x_id = grid[..., 0] // width
    grid_y_id = grid[..., 1] // height
    grid_ids = th.stack([grid_x_id, grid_y_id], dim=-1)
    dist_from_left = grid[..., 0] % width
    dist_from_right = width - dist_from_left
    dist_from_top = grid[..., 1] % height
    dist_from_bottom = height - dist_from_top
    dists = th.stack([dist_from_left, dist_from_right, dist_from_top, dist_from_bottom], dim=-1)
    min_dists, min_ids = th.min(dists, dim=-1)
    ### HACK
    # min_dists = min_dists/th.max(min_dists)
    # angel as well
    grid[..., 0] = grid[..., 0] % width
    grid[..., 1] = grid[..., 1] % height
    grid[..., 0] = grid[..., 0] / width
    grid[..., 1] = grid[..., 1] / height
    grid = grid * 2 - 1
    angle = th.atan2(grid[..., 1], grid[..., 0])
    new_grid = th.stack([min_dists, angle], dim=-1)
    return new_grid, grid_ids

def cart_to_brick_repeat_grid_y(grid, size, shift=None):
    width, height = _unpack_vec2(size)
    # similar to cartesian repeat, except a a width/2 shift in every alternate row
    if shift is None:
        shift = height / 2
    grid_x_id = grid[..., 0] // width
    grid[..., 1] = th.where(grid_x_id % 2 == 0, grid[..., 1] - shift, grid[..., 1])
    grid, grid_ids = rect_repeat_grid(grid, size)
    return grid, grid_ids

def cart_to_brick_repeat_grid_x(grid, size, shift=None):
    width, height = _unpack_vec2(size)
    # similar to cartesian repeat, except a a height/2 shift in every alternate row
    if shift is None:
        shift = width / 2
    grid_y_id = grid[..., 1] // height
    grid[..., 0] = th.where(grid_y_id % 2 == 0, grid[..., 0] - shift, grid[..., 0])
    grid, grid_ids = rect_repeat_grid(grid, size)
    return grid, grid_ids

def cart_to_brick_edge_grid_y(grid, size, shift=None):
    width, height = _unpack_vec2(size)
    if shift is None:
        shift = height / 2
    grid_x_id = grid[..., 0] // width
    grid[..., 1] = th.where(grid_x_id % 2 == 0, grid[..., 1] - shift, grid[..., 1])
    grid, grid_ids = rect_repeat_edge_grid(grid, size)
    return grid, grid_ids

def cart_to_brick_edge_grid_x(grid, size, shift=None):
    width, height = _unpack_vec2(size)
    if shift is None:
        shift = width / 2
    grid_y_id = grid[..., 1] // height
    grid[..., 0] = th.where(grid_y_id % 2 == 0, grid[..., 0] - shift, grid[..., 0])
    grid, grid_ids = rect_repeat_edge_grid(grid, size)
    return grid, grid_ids

def cart_to_hex_grid(grid, hex_size):
    ratio = 0.5 / np.sin(np.pi/3)
    grid = grid / (ratio * 2 * hex_size)
    device = grid.device
    S = _S.to(device)
    hC = th.floor(th.stack([grid, grid - _HEX_VAL_3.to(device)], dim=-1) / S[..., None]) + .5
    h = th.stack([grid - hC[..., 0] * S, grid - (hC[..., 1] + .5) * S], dim=-1)
    out_1 = th.stack([h[..., 0], hC[..., 0]], dim=-1)
    out_2 = th.stack([h[..., 1], hC[..., 1] + .5], dim=-1)
    condition = th.sum(h[..., 0]**2, dim=-1) < th.sum(h[..., 1]**2, dim=-1)
    condition = condition.unsqueeze(-1).unsqueeze(-1)
    output = th.where(condition, out_1, out_2)
    centroids = output[..., 1]
    centroids = centroids * 2
    grid = output[..., 0]
    grid_mult = hex_size * 1.268
    grid = grid / grid_mult
    # What if I want radial units? 
    return grid, centroids

def cart_to_hex_grid_flip(grid, hex_size):
    grid = grid.flip(-1)
    grid, grid_id = cart_to_hex_grid(grid, hex_size)
    grid_id = grid_id.flip(-1)
    return grid, grid_id

def cart_to_hex_edge_grid(grid, hex_size):
    grid, grid_id = cart_to_hex_grid(grid, hex_size)
    p = th.abs(grid)
    device = grid.device
    S = _S.to(device)
    edge_dist = th.maximum(th.sum(p * S * .5, dim=-1), p[..., 1])
    edge_dist = th.max(edge_dist) - edge_dist
    edge_dist = edge_dist * hex_size * 0.5 # Adjusting the size to be reasonable
    # min_dists = min_dists/th.max(min_dists)
    theta = th.atan2(grid[..., 1], grid[..., 0])
    grid = th.stack([edge_dist, theta], dim=-1)
    return grid, grid_id

def cart_to_hex_edge_grid_flip(grid, hex_size):
    grid = grid.flip(-1)
    grid, grid_id = cart_to_hex_edge_grid(grid, hex_size)
    return grid, grid_id


def cart_to_triangular_grid(grid, incircle_radius, center=True):
    if isinstance(incircle_radius, float):
        incircle_radius = th.tensor([incircle_radius]).to(grid.device)
    side_length = incircle_radius #  * np.sqrt(3)
    y_len = _SIN_PI_3 * side_length
    grid_size = th.stack([side_length/2, y_len], dim=1)# [None, ...]
    n_boxes = grid // grid_size
    local_coords = grid - n_boxes * grid_size
    col_id = n_boxes[..., 0]
    row_id = n_boxes[..., 1]
    delta_1 = th.where(local_coords[..., 1] > local_coords[..., 0] * _TAN_PI_3, 0, 1)
    delta_2 = th.where(local_coords[..., 1] > y_len - _TAN_PI_3 * local_coords[..., 0], 1, 0)
    col_shift = th.where(th.abs(col_id) % 2 == 0, delta_1, delta_2)
    n_boxes[..., 0] += col_shift
    local_coords[..., 0] -= col_shift * side_length / 2.0
    delta_1 = th.where(local_coords[..., 1] > local_coords[..., 0] * _TAN_PI_3, 1, 0)
    delta_2 = th.where(local_coords[..., 1] > y_len - _TAN_PI_3 * local_coords[..., 0], 1, 0)
    row_shift = th.where(th.abs(col_id) % 2 == 0, delta_1, delta_2)
    local_coords[..., 1] = th.where(row_shift == 0, local_coords[..., 1] , y_len - local_coords[..., 1]) 
    max_coord = th.max(local_coords)
    local_coords = local_coords / max_coord
    if center:
        local_coords[..., 1] -= 0.5
    return local_coords, n_boxes

def cart_to_triangular_edge_grid(grid, incircle_radius):
    if isinstance(incircle_radius, float):
        incircle_radius = th.tensor([incircle_radius]).to(grid.device)
    side_length = incircle_radius # * np.sqrt(3)
    orig_tri_grid, grid_id = cart_to_triangular_grid(grid, side_length)
    theta = th.atan2(orig_tri_grid[..., 1], orig_tri_grid[..., 0])
    
    y_len = _SIN_PI_3 * side_length
    grid_size = th.stack([side_length/2, y_len], dim=1)# [None, ...]
    n_boxes = grid // grid_size
    local_coords = grid - n_boxes * grid_size
    edge_dist_1 = local_coords[..., 1]
    rot_amount = np.pi / 3
    rot_mat = th.tensor([[np.cos(rot_amount), -np.sin(rot_amount)], [np.sin(rot_amount), np.cos(rot_amount)]]).to(grid.device).float()
    local_coords_2 = th.matmul(grid, rot_mat)
    n_boxes = local_coords_2 // grid_size
    local_coords_2 = local_coords_2 - n_boxes * grid_size
    edge_dist_2 = local_coords_2[..., 1]
    rot_amount = -np.pi / 3
    rot_mat = th.tensor([[np.cos(rot_amount), -np.sin(rot_amount)], [np.sin(rot_amount), np.cos(rot_amount)]]).to(grid.device).float()
    local_coords_3 = th.matmul(grid, rot_mat)
    n_boxes = local_coords_3 // grid_size
    local_coords_3 = local_coords_3 - n_boxes * grid_size
    edge_dist_3 = local_coords_3[..., 1]
    all_dist = th.stack([edge_dist_1, edge_dist_2, edge_dist_3,
                         y_len - edge_dist_1, y_len - edge_dist_2, y_len - edge_dist_3], dim=-1)
    min_dist, min_id = th.min(all_dist, dim=-1)
    new_grid = th.stack([min_dist, theta], dim=-1)
    return new_grid, grid_id
    
def cart_to_diamond_grid(grid, side_length, resize_dif=True):
    
    if isinstance(side_length, float):
        side_length = th.tensor([side_length]).to(grid.device)
    side_length = side_length * 2    
    y_len = _SIN_PI_3 * side_length
    grid_size = th.stack([side_length/2, y_len], dim=1)#[None, ...]
    n_boxes_1 = grid // grid_size
    # print('here', n_boxes_1[..., 0].min(), n_boxes_1[..., 0].max())
    local_coords = grid - n_boxes_1 * grid_size
    local_coords = local_coords - grid_size / 2
    # shifted 
    grid_shifted = grid - grid_size/2
    n_boxes_2 = grid_shifted // grid_size
    local_coords_2 = grid_shifted - n_boxes_2 * grid_size
    local_coords_2 = local_coords_2 - grid_size / 2
    abs_coord_1 = th.abs(local_coords)
    cond = (abs_coord_1[..., 1:2] < y_len/2 - _TAN_PI_3 * abs_coord_1[..., 0:1])
    real_coords = th.where(cond, local_coords, local_coords_2)
    if resize_dif:
        real_coords = real_coords / (grid_size[0, 0]/2 * 0.75)
    else:
        real_coords = real_coords / (grid_size[0, 0]/2)
    # print('here 2', n_boxes_2[..., 0].min(), n_boxes_2[..., 0].max())
    real_boxes = th.where(cond, n_boxes_1 * 2, 2 * n_boxes_2 + 1)
    real_boxes = (real_boxes + 1)
    return real_coords, real_boxes

def cart_to_diamond_edge_grid(grid, side_length):
    grid, grid_id = cart_to_diamond_grid(grid, side_length, resize_dif=False)
    abs_grid = th.abs(grid)
    y_len = _SIN_PI_3 * 2.0
    num = _TAN_PI_3 * abs_grid[..., 0] + abs_grid[..., 1] - y_len
    num = th.abs(num)
    edge_dist = num / DIAMOND_DENOM * side_length * 0.5
    theta = th.atan2(grid[..., 1], grid[..., 0])
    grid = th.stack([edge_dist, theta], dim=-1) 
    return grid, grid_id


    
def cartesian_repeat_x_grid(grid, width):
    # -1 to 1 repeating
    # This should also return assignments
    grid_x_id = grid[..., 0] // width
    grid_x_id = grid_x_id.unsqueeze(-1)
    grid[..., 0] = grid[..., 0] % width
    grid[..., 0] = grid[..., 0] / width
    grid[..., 0] = grid[..., 0] * 2 - 1
    return grid, grid_x_id

def cartesian_repeat_y_grid(grid, height):
    # -1 to 1 repeating
    # This should also return assignments
    grid_y_id = grid[..., 1] // height
    grid_y_id = grid_y_id.unsqueeze(-1)
    grid[..., 1] = grid[..., 1] % height
    grid[..., 1] = grid[..., 1] / height
    grid[..., 1] = grid[..., 1] * 2 - 1
    return grid, grid_y_id


def polar_repeat_angular_grid(grid, angular_unit):
    theta_id = grid[..., 1:2] // angular_unit
    grid[..., 1] = grid[..., 1] % angular_unit
    # center it
    grid[..., 1] = grid[..., 1] - angular_unit / 2.0
    return grid, theta_id

def RadialRepeatCentered(grid, radial_unit):
    radial_id = grid[..., 0:1] // radial_unit
    grid[..., 0] = grid[..., 0] % radial_unit
    # center it
    return grid, radial_id

def polar_repeat_centered_grid(grid, cell_size):
    radial_unit, angular_unit = _unpack_vec2(cell_size)
    radial_id = grid[..., 0] // radial_unit
    angular_id = grid[..., 1] // angular_unit
    grid_id = th.stack([radial_id, angular_id], dim=-1)
    
    center_r = radial_id * radial_unit + radial_unit / 2.0
    center_theta = angular_id * angular_unit + angular_unit / 2.0
    center_x = center_r * th.cos(center_theta)
    center_y = center_r * th.sin(center_theta)
    cart_x = grid[..., 0] * th.cos(grid[..., 1])
    cart_y = grid[..., 0] * th.sin(grid[..., 1])
    vec_from_center_x = cart_x - center_x
    vec_from_center_y = cart_y - center_y
    inner_radius = th.sqrt(vec_from_center_x**2 + vec_from_center_y**2)
    angle = th.atan2(vec_from_center_y, vec_from_center_x) - grid[..., 1]
    angle = angle % (2 * np.pi)
    new_grid = th.stack([inner_radius, angle], dim=-1) 
    return new_grid, grid_id

def polar_repeat_init_radial_grid(grid, cell_size, init_gap):
    radial_unit, angular_unit = _unpack_vec2(cell_size)
    grid[..., 1] += th.pi
    grid[..., 0]= grid[..., 0] + init_gap
    radial_id = grid[..., 0] // radial_unit
    angular_id = grid[..., 1] // angular_unit
    grid_id = th.stack([radial_id, angular_id], dim=-1)
    grid[..., 0] = grid[..., 0] % radial_unit
    grid[..., 0] = (grid[..., 0] - radial_unit /2 ) / (radial_unit/2.0)
    grid_partitioned = grid[..., 1] % angular_unit
    grid_partitioned = (grid_partitioned - angular_unit / 2.0) / (angular_unit / 2.0)
    grid[..., 1] = grid_partitioned

    pi_ratio = np.pi / angular_unit
    # if th.round(pi_ratio % 2) != 0:
    #     grid_id[:, 1][grid_id[:, 1]==grid_id[:, 1].min()] = grid_id[:, 1].max() # only if 
    # center it
    return grid, grid_id


def polar_repeat_bricked_grid(grid, cell_size, init_gap):
    radial_unit, angular_unit = _unpack_vec2(cell_size)
    grid[..., 1] += th.pi
    grid[..., 0]= grid[..., 0] + init_gap
    radial_id = grid[..., 0] // radial_unit
    mask = radial_id % 2 == 0
    grid[..., 1] = (grid[..., 1] + mask * angular_unit / 2.0) % (2 * th.pi)
    angular_id = grid[..., 1] // angular_unit

    grid_id = th.stack([radial_id, angular_id], dim=-1)
    grid[..., 0] = grid[..., 0] % radial_unit
    grid[..., 0] = (grid[..., 0] - radial_unit /2 ) / (radial_unit/2.0)
    grid_partitioned = grid[..., 1] % angular_unit
    grid_partitioned = (grid_partitioned - angular_unit / 2.0) / (angular_unit / 2.0)
    grid[..., 1] = grid_partitioned

    # pi_ratio = np.pi / angular_unit
    # if th.round(pi_ratio % 2) != 0:
    #     grid_id[:, 1][grid_id[:, 1]==grid_id[:, 1].min()] = grid_id[:, 1].max() # only if 
    # center it
    return grid, grid_id

def polar_repeat_radial_fixed_arc_grid(grid, cell_size, init_gap):
    radial_unit, arc_length = _unpack_vec2(cell_size)
    grid[..., 0]= grid[..., 0] + init_gap
    grid[..., 1] += th.pi
    radial_id = grid[..., 0] // radial_unit
    total_arc_size = 2 * np.pi * (radial_id + 1) * radial_unit

    angular_units = th.round(total_arc_size / arc_length)
    angle_divider = 2 * np.pi / (angular_units + 1e-9)


    # mark_neg = (angular_units % 2 == 1)
    # grid[..., 1] = grid[..., 1] + mark_neg  * np.pi # (angle_divider/2.0)
    angular_id = grid[..., 1] // angle_divider

    grid_id = th.stack([radial_id, angular_id], dim=-1)
    grid[..., 0] = (grid[..., 0] % radial_unit - radial_unit /2) / (radial_unit/2.0)
    grid[..., 1] = grid[..., 1] % angle_divider
    # center it
    grid[..., 1] = (grid[..., 1] - angle_divider / 2.0) / (angle_divider / 2.0)
    return grid, grid_id


def polar_repeat_fixed_arc_bricked_grid(grid, cell_size, init_gap):
    radial_unit, arc_length = _unpack_vec2(cell_size)
    grid[..., 0]= grid[..., 0] + init_gap
    grid[..., 1] += th.pi
    
    radial_id = grid[..., 0] // radial_unit

    total_arc_size = 2 * np.pi * (radial_id + 1) * radial_unit

    angular_units = th.round(total_arc_size / arc_length)
    angle_divider = 2 * np.pi / (angular_units + 1e-9)

    mask = radial_id % 2 == 0
    grid[..., 1] = (grid[..., 1] + mask * angle_divider / 2.0) % (2 * th.pi)
    angular_id = grid[..., 1] // angle_divider

    grid_id = th.stack([radial_id, angular_id], dim=-1)
    
    grid[..., 0] = (grid[..., 0] % radial_unit - radial_unit /2) / (radial_unit/2.0)
    grid[..., 1] = grid[..., 1] % angle_divider
    # center it
    grid[..., 1] = (grid[..., 1] - angle_divider / 2.0) / (angle_divider / 2.0)
    return grid, grid_id


def polar_repeat_edge_grid(grid, cell_size, init_gap):
    radial_unit, angular_unit = _unpack_vec2(cell_size)
    # Assume input to be a polar grid itself
    # radial distance
    # Go from 0 to 2 pi instead of -1 to 1 pi
    grid[..., 0]= grid[..., 0] + init_gap
    grid[..., 1] += th.pi


    radial_lower_bound = grid[..., 0] % radial_unit
    radial_upper_bound = radial_unit - radial_lower_bound
    # distance from angular edges -> use r * theta
    angular_lower_bound = grid[..., 1] % angular_unit
    angular_upper_bound = angular_unit - angular_lower_bound
    angular_dist_lower = grid[..., 0] * angular_lower_bound
    angular_dist_upper = grid[..., 0] * angular_upper_bound
    dists = th.stack([radial_lower_bound, radial_upper_bound, angular_dist_lower, angular_dist_upper], dim=-1)
    min_dists, min_ids = th.min(dists, dim=-1)
    grid_radial_id = grid[..., 0] // radial_unit
    grid_angular_id = grid[..., 1] // angular_unit
    grid_id = th.stack([grid_radial_id, grid_angular_id], dim=-1)

    center_r = grid_radial_id * radial_unit + radial_unit / 2.0
    center_theta = grid_angular_id * angular_unit + angular_unit / 2.0
    center_x = center_r * th.cos(center_theta)
    center_y = center_r * th.sin(center_theta)
    # for each point get the cart
    cart_x = grid[..., 0] * th.cos(grid[..., 1])
    cart_y = grid[..., 0] * th.sin(grid[..., 1])
    vec_from_center_x = cart_x - center_x
    vec_from_center_y = cart_y - center_y
    angle = th.atan2(vec_from_center_y, vec_from_center_x)
    real_angle = angle - grid[..., 1]
    real_angle = real_angle % (2 * np.pi)
    new_grid = th.stack([min_dists, real_angle], dim=-1)

    # pi_ratio = np.pi / angular_unit
    # if th.round(pi_ratio % 2) != 0:
    #     grid_id[:, 1][grid_id[:, 1]==grid_id[:, 1].min()] = grid_id[:, 1].max() # only if 

    return new_grid, grid_id

def polar_repeat_edge_bricked_grid(grid, cell_size, init_gap):
    radial_unit, angular_unit = _unpack_vec2(cell_size)
    # Assume input to be a polar grid itself
    # radial distance
    grid[..., 0]= grid[..., 0] + init_gap
    grid[..., 1] += th.pi

    radial_id = grid[..., 0] // radial_unit
    mask = radial_id % 2 == 0
    grid[..., 1] = (grid[..., 1] + mask * angular_unit / 2.0) % (2 * th.pi)

    radial_lower_bound = grid[..., 0] % radial_unit
    radial_upper_bound = radial_unit - radial_lower_bound
    # distance from angular edges -> use r * theta
    angular_lower_bound = grid[..., 1] % angular_unit
    angular_upper_bound = angular_unit - angular_lower_bound
    angular_dist_lower = grid[..., 0] * angular_lower_bound
    angular_dist_upper = grid[..., 0] * angular_upper_bound
    dists = th.stack([radial_lower_bound, radial_upper_bound, angular_dist_lower, angular_dist_upper], dim=-1)
    min_dists, min_ids = th.min(dists, dim=-1)
    grid_radial_id = grid[..., 0] // radial_unit
    grid_angular_id = grid[..., 1] // angular_unit
    grid_id = th.stack([grid_radial_id, grid_angular_id], dim=-1)

    center_r = grid_radial_id * radial_unit + radial_unit / 2.0
    center_theta = grid_angular_id * angular_unit + angular_unit / 2.0
    center_x = center_r * th.cos(center_theta)
    center_y = center_r * th.sin(center_theta)
    # for each point get the cart
    cart_x = grid[..., 0] * th.cos(grid[..., 1])
    cart_y = grid[..., 0] * th.sin(grid[..., 1])
    vec_from_center_x = cart_x - center_x
    vec_from_center_y = cart_y - center_y
    angle = th.atan2(vec_from_center_y, vec_from_center_x)
    real_angle = angle - grid[..., 1]
    real_angle = real_angle % (2 * np.pi)
    new_grid = th.stack([min_dists, real_angle], dim=-1)

    # pi_ratio = np.pi / angular_unit
    # if th.round(pi_ratio % 2) != 0:
    #     grid_id[:, 1][grid_id[:, 1]==grid_id[:, 1].min()] = grid_id[:, 1].max() # only if 

    return new_grid, grid_id

def polar_repeat_edge_fixed_arc(grid, cell_size, init_gap):
    radial_unit, arc_length = _unpack_vec2(cell_size)
    # Assume input to be a polar grid itself
    # radial distance
    grid[..., 0]= grid[..., 0] + init_gap
    grid[..., 1] += th.pi
    
    radial_lower_bound = grid[..., 0] % radial_unit
    radial_upper_bound = radial_unit - radial_lower_bound
    # distance from angular edges -> use r * theta

    radial_id = grid[..., 0] // radial_unit
    total_arc_size = 2 * np.pi * (radial_id + 1) * radial_unit

    angular_units = th.round(total_arc_size / arc_length)
    angle_divider = 2 * np.pi / (angular_units + 1e-9)

    angular_lower_bound = grid[..., 1] % angle_divider
    angular_upper_bound = angle_divider - angular_lower_bound
    angular_dist_lower = grid[..., 0] * angular_lower_bound
    angular_dist_upper = grid[..., 0] * angular_upper_bound
    dists = th.stack([radial_lower_bound, radial_upper_bound, angular_dist_lower, angular_dist_upper], dim=-1)
    min_dists, min_ids = th.min(dists, dim=-1)
    grid_radial_id = grid[..., 0] // radial_unit
    grid_angular_id = grid[..., 1] // angle_divider
    grid_id = th.stack([grid_radial_id, grid_angular_id], dim=-1)

    center_r = grid_radial_id * radial_unit + radial_unit / 2.0
    center_theta = grid_angular_id * angle_divider + angle_divider / 2.0
    center_x = center_r * th.cos(center_theta)
    center_y = center_r * th.sin(center_theta)
    # for each point get the cart
    cart_x = grid[..., 0] * th.cos(grid[..., 1])
    cart_y = grid[..., 0] * th.sin(grid[..., 1])
    vec_from_center_x = cart_x - center_x
    vec_from_center_y = cart_y - center_y
    angle = th.atan2(vec_from_center_y, vec_from_center_x)
    real_angle = angle - grid[..., 1]
    real_angle = real_angle % (2 * np.pi)
    new_grid = th.stack([min_dists, real_angle], dim=-1)
    return new_grid, grid_id

def polar_repeat_edge_fixed_arc_bricked(grid, cell_size, init_gap):
    radial_unit, arc_length = _unpack_vec2(cell_size)
    # Assume input to be a polar grid itself
    # radial distance
    grid[..., 0]= grid[..., 0] + init_gap
    grid[..., 1] += th.pi
    
    radial_lower_bound = grid[..., 0] % radial_unit
    radial_upper_bound = radial_unit - radial_lower_bound
    # distance from angular edges -> use r * theta

    radial_id = grid[..., 0] // radial_unit
    total_arc_size = 2 * np.pi * (radial_id + 1) * radial_unit

    angular_units = th.round(total_arc_size / arc_length)
    angle_divider = 2 * np.pi / (angular_units + 1e-9)

    mask = radial_id % 2 == 0
    grid[..., 1] = (grid[..., 1] + mask * angle_divider / 2.0) % (2 * th.pi)

    angular_lower_bound = grid[..., 1] % angle_divider
    angular_upper_bound = angle_divider - angular_lower_bound
    angular_dist_lower = grid[..., 0] * angular_lower_bound
    angular_dist_upper = grid[..., 0] * angular_upper_bound
    dists = th.stack([radial_lower_bound, radial_upper_bound, angular_dist_lower, angular_dist_upper], dim=-1)
    min_dists, min_ids = th.min(dists, dim=-1)
    grid_radial_id = grid[..., 0] // radial_unit
    grid_angular_id = grid[..., 1] // angle_divider
    grid_id = th.stack([grid_radial_id, grid_angular_id], dim=-1)

    center_r = grid_radial_id * radial_unit + radial_unit / 2.0
    center_theta = grid_angular_id * angle_divider + angle_divider / 2.0
    center_x = center_r * th.cos(center_theta)
    center_y = center_r * th.sin(center_theta)
    # for each point get the cart
    cart_x = grid[..., 0] * th.cos(grid[..., 1])
    cart_y = grid[..., 0] * th.sin(grid[..., 1])
    vec_from_center_x = cart_x - center_x
    vec_from_center_y = cart_y - center_y
    angle = th.atan2(vec_from_center_y, vec_from_center_x)
    real_angle = angle - grid[..., 1]
    real_angle = real_angle % (2 * np.pi)
    new_grid = th.stack([min_dists, real_angle], dim=-1)
    return new_grid, grid_id
def get_low_disc_centroids(grid, n_centroids_x, n_centroids_y, noise_rate):
    min_x = th.min(grid[..., 0])
    max_x = th.max(grid[..., 0])
    min_y = th.min(grid[..., 1])
    max_y = th.max(grid[..., 1])
    if isinstance(n_centroids_x, th.Tensor):
        n_centroids_x = n_centroids_x.long().item()
        n_centroids_y = n_centroids_y.long().item()
    else:
        n_centroids_x = int(n_centroids_x)
        n_centroids_y = int(n_centroids_y)
    grid_size = th.stack([(max_x - min_x) / n_centroids_x, (max_y - min_y) / n_centroids_y], dim=0) 
    mesh_grid = th.meshgrid(th.linspace(min_x - grid_size[0], max_x + grid_size[0], n_centroids_y + 2), 
                                th.linspace(min_y - grid_size[1], max_y + grid_size[1], n_centroids_x + 2))
    mesh_grid = th.stack(mesh_grid, dim=-1)
    mesh_grid = mesh_grid.reshape(-1, 2)
    uniform_shifts = noise_rate * th.rand((n_centroids_x + 2) * (n_centroids_y + 2), 2).to(grid.device) * grid_size
    mesh_grid = mesh_grid.to(grid.device)
    centroids = mesh_grid + uniform_shifts
    return centroids

def get_n_centroids(grid, x_size, y_size):

    grid_x = grid[..., 0]
    grid_y = grid[..., 1]
    grid_x_min, grid_x_max, grid_y_min, grid_y_max = grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()
    n_centroids_x = th.ceil((grid_x_max - grid_x_min) / x_size)
    n_centroids_y = th.ceil((grid_y_max - grid_y_min) / y_size)
    n_centroids = n_centroids_x * n_centroids_y
    if n_centroids > CENTROIDS_UPPER_LIMIT:
        # reduce by ratio
        ratio = np.sqrt(CENTROIDS_UPPER_LIMIT / n_centroids.item())
        n_centroids_x = th.clamp(th.floor(n_centroids_x * ratio), min=4)
        n_centroids_y = th.clamp(th.floor(n_centroids_y * ratio), min=4)
    # make sure it is at least 1
    return n_centroids_x, n_centroids_y
    
# Voronoi -> Center with the delaunay triangle centers
def cart_to_voronoi_grid_sizewise(grid, cell_size, noise_rate=0.5, precomputed_centroids=None):
    # use traditional size
    x_size, y_size = _unpack_vec2(cell_size)
    n_centroids_x, n_centroids_y = get_n_centroids(grid, x_size, y_size)
    return cart_to_voronoi_grid(grid, n_centroids_x, n_centroids_y, noise_rate, precomputed_centroids)

def cart_to_voronoi_grid(grid, n_centroids_x, n_centroids_y, noise_rate=0.5, precomputed_centroids=None):
    
    if precomputed_centroids is None:
        centroids = get_low_disc_centroids(grid, n_centroids_x, n_centroids_y, noise_rate)
    else:
        centroids = precomputed_centroids
    # Make a meshgrid of centroids
    assigned_centroids = th.argmin(th.norm(grid[:, None, :] - centroids[None, :, :], dim=-1), dim=-1)
    
    grid = grid - centroids[assigned_centroids]
    grid_id_y = assigned_centroids % (n_centroids_x + 2)
    grid_id_x = assigned_centroids // (n_centroids_x + 2)
    grid_id = th.stack([grid_id_x, grid_id_y], dim=-1)
    return grid, grid_id

def cart_to_voronoi_radially_deformed_grid(grid, cell_size, noise_rate=0.5, precomputed_centroids=None):

    x_size, y_size = _unpack_vec2(cell_size)
    n_centroids_x, n_centroids_y = get_n_centroids(grid, x_size, y_size)

    if precomputed_centroids is None:
        centroids = get_low_disc_centroids(grid, n_centroids_x, n_centroids_y, noise_rate)
    else:
        centroids = precomputed_centroids

    distances = th.norm(grid[:, None, :] - centroids[None, :, :], dim=-1)
    # Make voronoi dist using F2 - F1
    min_dists, closest = th.topk(distances, 2, dim=1, largest=False)
    relative_coords_1 = grid - th.gather(centroids, 0, closest[:, 0].unsqueeze(1).expand(-1, 2))
    relative_coords_2 = grid - th.gather(centroids, 0, closest[:, 1].unsqueeze(1).expand(-1, 2))
    radius = relative_coords_1.norm(dim=-1)/relative_coords_2.norm(dim=-1)

    theta = th.atan2(relative_coords_1[:, 1], relative_coords_1[:, 0])
    theta = theta / np.pi
    centroid_assignments = closest[:, 0:1]
    grid_id_y = centroid_assignments % (n_centroids_x + 2)
    grid_id_x = centroid_assignments // (n_centroids_x + 2)
    grid_id = th.cat([grid_id_x, grid_id_y], dim=-1)
    grid = th.stack([radius, theta], dim=-1)
    return grid, grid_id

def cart_to_voronoi_edge_grid_sizewise(grid, cell_size, noise_rate=0.5, precomputed_centroids=None):

    x_size, y_size = _unpack_vec2(cell_size)
    n_centroids_x, n_centroids_y = get_n_centroids(grid, x_size, y_size)

    return cart_to_voronoi_edge_grid(grid, n_centroids_x, n_centroids_y, noise_rate, precomputed_centroids)


def cart_to_voronoi_edge_grid(grid, n_centroids_x, n_centroids_y, noise_rate=0.5, precomputed_centroids=None):
    # Make a meshgrid of centroids
    if precomputed_centroids is None:
        centroids = get_low_disc_centroids(grid, n_centroids_x, n_centroids_y, noise_rate)
    else:
        centroids = precomputed_centroids

    distances = th.norm(grid[:, None, :] - centroids[None, :, :], dim=-1)
    # Make voronoi dist using F2 - F1
    min_dists, closest = th.topk(distances, 2, dim=1, largest=False)
    relative_coords_1 = grid - th.gather(centroids, 0, closest[:, 0].unsqueeze(1).expand(-1, 2))
    relative_coords_2 = grid - th.gather(centroids, 0, closest[:, 1].unsqueeze(1).expand(-1, 2))
    radius = relative_coords_2.norm(dim=-1) - relative_coords_1.norm(dim=-1)

    theta = th.atan2(relative_coords_1[:, 1], relative_coords_1[:, 0])
    centroid_assignments = closest[:, 0:1]
    centroid_assignments = closest[:, 0:1]
    grid_id_y = centroid_assignments % (n_centroids_x + 2)
    grid_id_x = centroid_assignments // (n_centroids_x + 2)
    grid_id = th.cat([grid_id_x, grid_id_y], dim=-1)
    grid = th.stack([radius, theta], dim=-1)
    return grid, grid_id

# wang tiles
def get_precomputed_n_vals(grid, box_height, box_width, noise_rate):
    # box_height/box_width kept as scalars here; pass as a Vec2 tuple
    grid, grid_ids = rect_repeat_grid(grid, (box_width, box_height))
    checkerboard_ids = grid_ids.sum(-1) % 2

    unique_ids, inverse_indices = grid_ids.unique(dim=0, return_inverse=True)
    num_unique_ids = unique_ids.size(0)
    tuple_ids = [tuple(list(x.cpu().numpy())) for x in unique_ids]#  unique_ids.cpu().numpy()))
    n_vals = th.rand((num_unique_ids), device=grid.device) * noise_rate
    return n_vals

def cart_to_aperiodic_box_grid_ids(grid, box_size, noise_rate=0.5, precomputed_n_vals=None):
    box_height, box_width = _unpack_vec2(box_size)
    grid, grid_ids = rect_repeat_grid(grid, (box_width, box_height))
    checkerboard_ids = grid_ids.sum(-1) % 2

    unique_ids, inverse_indices = grid_ids.unique(dim=0, return_inverse=True)
    num_unique_ids = unique_ids.size(0)
    tuple_ids = [tuple(list(x.cpu().numpy())) for x in unique_ids]#  unique_ids.cpu().numpy()))
    if precomputed_n_vals is not None:
        n_vals = precomputed_n_vals
    else:
        n_vals = th.rand((num_unique_ids), device=grid.device) * noise_rate

    half_tensor = th.tensor(0.5, device=grid.device)
    tuple_to_val = defaultdict(lambda: half_tensor)
    temp = {tuple_id: n_vals[i] for i, tuple_id in enumerate(tuple_ids)}

    tuple_to_val.update(temp)
    x_pos_ids = unique_ids.clone()
    x_pos_ids[:, 0] += 1
    x_neg_ids = unique_ids.clone()
    x_neg_ids[:, 0] -= 1
    y_pos_ids = unique_ids.clone()
    y_pos_ids[:, 1] += 1
    y_neg_ids = unique_ids.clone()
    y_neg_ids[:, 1] -= 1

    x_pos_vals = [tuple_to_val[tuple(list(x.cpu().numpy()))] for x in x_pos_ids]
    x_neg_vals = [tuple_to_val[tuple(list(x.cpu().numpy()))] for x in x_neg_ids]
    y_pos_vals = [tuple_to_val[tuple(list(x.cpu().numpy()))] for x in y_pos_ids]
    y_neg_vals = [tuple_to_val[tuple(list(x.cpu().numpy()))] for x in y_neg_ids]

    pp_n_vals = n_vals[inverse_indices]
    pp_x_pos_vals = th.tensor(x_pos_vals, device=grid.device)[inverse_indices]
    pp_x_neg_vals = th.tensor(x_neg_vals, device=grid.device)[inverse_indices]
    pp_y_pos_vals = th.tensor(y_pos_vals, device=grid.device)[inverse_indices]
    pp_y_neg_vals = th.tensor(y_neg_vals, device=grid.device)[inverse_indices]

    # Now path 1 -> n is a y val, and up down left right are x vals.
    grid = (grid + 1)/2

    y_grid = grid_ids.clone()
    c1 = grid[:, 1] > pp_n_vals
    c2 = grid[:, 0] > pp_y_pos_vals
    c3 = grid[:, 0] > pp_y_neg_vals
    # next_grid_ids = grid_ids.clone() + 1
    y_grid[:, 1] = th.where(c1, grid_ids[:, 1] + 1, grid_ids[:, 1])
    y_grid[:, 0] = th.where(c1 & c2, grid_ids[:, 0]+1, 
                                grid_ids[:, 0])
    y_grid[:, 0] = th.where((~c1) & c3, grid_ids[:, 0] + 1, 
                                y_grid[:, 0])

    x_grid = grid_ids.clone()
    c1 = grid[:, 0] > pp_n_vals
    c2 = grid[:, 1] > pp_x_pos_vals
    c3 = grid[:, 1] > pp_x_neg_vals
    x_grid[:, 0] = th.where(c1, grid_ids[:, 0] + 1, grid_ids[:, 0])
    x_grid[:, 1] = th.where(c1 & c2, grid_ids[:, 1]+1, 
                                grid_ids[:, 1])
    x_grid[:, 1] = th.where((~c1) & c3, grid_ids[:, 1]+1,
                                    x_grid[:, 1])

    new_grid_ids = th.where(checkerboard_ids.bool().unsqueeze(-1), x_grid, y_grid)
    # Now this needs to be used for renorm
    return new_grid_ids

def cart_to_aperiodic_box_grid(grid, simple_grid, box_size, noise_rate=0.5, precomputed_n_vals=None):
    grid_ids = cart_to_aperiodic_box_grid_ids(grid, box_size, noise_rate, precomputed_n_vals)
    new_grid = voronoi_style_normalize(grid_ids, simple_grid, simple_grid)
    return new_grid, grid_ids

def cart_to_aperiodic_box_edge_grid(grid, simple_grid, box_size, noise_rate=0.5, precomputed_n_vals=None):
    grid_ids = cart_to_aperiodic_box_grid_ids(grid, box_size, noise_rate, precomputed_n_vals)
    min_grid_x, max_grid_x = get_binwise_min_max(grid_ids, simple_grid[..., 0])
    min_grid_y, max_grid_y = get_binwise_min_max(grid_ids, simple_grid[..., 1])
    x_below_dist = grid[..., 0] - min_grid_x
    x_above_dist = max_grid_x - grid[..., 0]
    y_below_dist = grid[..., 1] - min_grid_y
    y_above_dist = max_grid_y - grid[..., 1]
    all_dist = th.stack([x_below_dist, x_above_dist, y_below_dist, y_above_dist], dim=-1)
    min_dists = th.min(all_dist, dim=-1, keepdim=True)[0]
    angle = th.atan2(grid[..., 1:2], grid[..., 0:1])
    grid_edge = th.cat([min_dists, angle], dim=-1)
    return grid_edge, grid_ids

# Calculate barycentric coordinates for each point w.r.t its associated Delaunay triangle
def compute_barycentric_coords(triangles, points):
    # triangles: (N, 3, 2)
    # points: (N, 2)
    A = triangles[:, 0, :]
    B = triangles[:, 1, :]
    C = triangles[:, 2, :]
    
    # Compute vectors
    v0 = B - A
    v1 = C - A
    v2 = points - A

    # Compute dot products
    d00 = th.sum(v0 * v0, dim=1)
    d01 = th.sum(v0 * v1, dim=1)
    d11 = th.sum(v1 * v1, dim=1)
    d20 = th.sum(v2 * v0, dim=1)
    d21 = th.sum(v2 * v1, dim=1)

    # Compute denominators
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    return th.stack([u, v, w], dim=-1)

def cart_to_delaunay_grid(grid, cell_size, noise_rate=0.5, precomputed_centroids=None):
    # Compute centroids (Voronoi cell centers)
    n_centroids_x, n_centroids_y = _unpack_vec2(cell_size)
    n_centroids_x = int(n_centroids_x)
    n_centroids_y = int(n_centroids_y)
    if precomputed_centroids is None:
        centroids = get_low_disc_centroids(grid, n_centroids_x, n_centroids_y, noise_rate)
    else:
        centroids = precomputed_centroids

    # Create a Delaunay triangulation of the centroids
    delaunay_triangulation = Delaunay(centroids.cpu().numpy())
    
    # Find the Delaunay triangle each point in the grid belongs to
    simplex_indices = delaunay_triangulation.find_simplex(grid.cpu().numpy())
    
    # Get the vertices of the corresponding Delaunay triangles
    delaunay_vertices = th.tensor(delaunay_triangulation.simplices, dtype=th.long, device=grid.device)
    delaunay_triangles = centroids[delaunay_vertices]  # Shape: (n_triangles, 3, 2)

    # Get the Delaunay triangle each grid point is associated with
    associated_triangles = delaunay_triangles[simplex_indices]  # Shape: (N, 3, 2)


    triangle_centers = th.mean(associated_triangles, dim=1)
    grid = grid - triangle_centers
    # Compute barycentric coordinates for all points
    # bary_coords = compute_barycentric_coords(associated_triangles, grid)
    # Convert bary centric to 2D grid with the center at the centroid.

    
    # Combine barycentric coordinates with the triangle vertices to get the new grid
    # grid_transformed = th.einsum('ij,ijk->ik', bary_coords, associated_triangles)

    simplex_indices = th.tensor(simplex_indices[..., None], dtype=th.long, device=grid.device)
    # Return the transformed grid and the associated triangle indices (for reference)
    return grid, simplex_indices

# Variations
def cart_translate_grid(grid, param):
    if isinstance(param, th.Tensor) and param.device != grid.device:
        param = param.to(grid.device, dtype=grid.dtype)
    elif not isinstance(param, th.Tensor):
        param = th.tensor(param, device=grid.device, dtype=grid.dtype)
    grid = grid + param[None, ...]
    return grid

def cart_scale_grid(grid, param):
    if not isinstance(param, th.Tensor):
        param = th.tensor(param, device=grid.device, dtype=grid.dtype)
    elif param.device != grid.device:
        param = param.to(grid.device, dtype=grid.dtype)
    grid = grid / param[None, ...]
    return grid

def cart_rotate_grid(grid, theta):
    if not isinstance(theta, th.Tensor):
        theta = th.tensor(theta, device=grid.device, dtype=grid.dtype)
    elif theta.device != grid.device:
        theta = theta.to(grid.device, dtype=grid.dtype)
    c, s = th.cos(theta), th.sin(theta)
    rot_mat = th.stack([c, -s, s, c], dim=0).reshape(2, 2)
    grid = th.matmul(grid, rot_mat)
    return grid

def cart_affine_grid(grid, transform):
    # add a column of ones
    ones = th.ones_like(grid[..., 0])
    grid = th.stack([grid[..., 0], grid[..., 1], ones], dim=-1)
    # apply the affine transform
    grid = th.matmul(grid, transform)   
    return grid[..., :2]

def polar_rotate_grid(grid, theta):
    grid[..., 1] = grid[..., 1] + theta
    return grid

def polar_scale_grid(grid, param):
    grid[..., 0] = grid[..., 0] * param
    return grid

def polar_translate_grid(grid, param):
    cart_x = grid[..., 0] * th.cos(grid[..., 1])
    cart_y = grid[..., 0] * th.sin(grid[..., 1])
    cart_x = cart_x + param[0]
    cart_y = cart_y + param[1]
    grid[..., 0] = th.sqrt(cart_x**2 + cart_y**2)
    grid[..., 1] = th.atan2(cart_y, cart_x)
    return grid

## Deformation

def scale_with_signal(grid, signal, apply_exp=False, invert=False, scaling_sigma=1):
    if apply_exp:
        signal = th.exp(-signal/scaling_sigma)
    if invert:
        signal = 1 - signal
    # Flatten signal to 1D before expanding, in case it's already (N, 1)
    if signal.dim() > 1:
        signal = signal.squeeze(-1)
    grid = grid * signal[:, None]
    return grid

def rotate_with_signal(grid, signal, apply_exp=False, invert=False, scaling_sigma=1):
    if apply_exp:
        signal = th.exp(-signal/scaling_sigma)
    if invert:
        signal = 1 - signal
    coses = th.cos(signal)
    sines = th.sin(signal)
    rot_mat_row_1 = th.stack([coses, -sines], dim=-1)
    rot_mat_row_2 = th.stack([sines, coses], dim=-1)
    rot_mat = th.stack([rot_mat_row_1, rot_mat_row_2], dim=-1)
    grid = th.bmm(grid[:, None, :], rot_mat)
    grid = grid.squeeze(1)
    return grid

def translate_x_with_signal(grid, signal, apply_exp=False, invert=False, scaling_sigma=1):
    if apply_exp:
        signal = th.exp(-signal/scaling_sigma)
    if invert:
        signal = 1 - signal
    grid[..., 0] = grid[..., 0] + signal
    return grid

def translate_y_with_signal(grid, signal, apply_exp=True, invert=False, scaling_sigma=1):
    if apply_exp:
        signal = th.exp(-signal/scaling_sigma)
    if invert:
        signal = 1 - signal
    grid[..., 1] = grid[..., 1] + signal
    return grid

def scale_x_with_signal(grid, signal, apply_exp=False, invert=False, scaling_sigma=1):
    if apply_exp:
        signal = th.exp(-signal/scaling_sigma)
    if invert:
        signal = 1 - signal
    grid[..., 0] = grid[..., 0] * signal
    return grid

def scale_y_with_signal(grid, signal, apply_exp=False, invert=False, scaling_sigma=1):
    if apply_exp:
        signal = th.exp(-signal/scaling_sigma)
    if invert:
        signal = 1 - signal
    grid[..., 1] = grid[..., 1] * signal
    return grid

def translate_with_signal(grid, signal, apply_exp=False, invert=False, scaling_sigma=1):
    if apply_exp:
        signal = th.exp(-signal/scaling_sigma)
    if invert:
        signal = 1 - signal
    grid = grid + signal
    return grid


# value noise
# gradient noise
# multilevel noise

def value_noise(resolution, noise_res):
    # generate noise
    if not isinstance(noise_res, th.Tensor):
        noise_res = th.tensor(noise_res)
    device = noise_res.device
    grid = th.meshgrid(th.linspace(0, resolution, resolution, device=device), 
                       th.linspace(0, resolution, resolution, device=device))
    grid = th.stack(grid, dim=-1).reshape(-1, 2)
    
    grid_ind = grid // noise_res
    grid_ind = grid_ind.long()
    factors = (grid % noise_res) / noise_res
    # sample the vaues based on the 
    n_noise = int(th.ceil(resolution / noise_res).item()) + 2
    noise_grid = th.rand((n_noise, n_noise), device=device)
    # # -> let the floor, the second and the third
    a = noise_grid[grid_ind[:, 0], grid_ind[:, 1]]
    b = noise_grid[grid_ind[:, 0] + 1, grid_ind[:, 1]]
    c = noise_grid[grid_ind[:, 0], grid_ind[:, 1] + 1]
    d = noise_grid[grid_ind[:, 0] + 1, grid_ind[:, 1] + 1]
    # bilinear interpolation
    b_1 = th.lerp(b, a, 1 - factors[:, 0])
    b_2 = th.lerp(d, c, 1 - factors[:, 0])
    noise = th.lerp(b_2, b_1, 1 - factors[:, 1])
    return noise

def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def lerp(a, b, t):
    return a + t * (b - a)

def gradient(h, x, y, vectors):
    g = vectors[h % 8]
    return g[:, 0] * x + g[:, 1] * y

def perlin_noise(resolution, noise_res):
    if not isinstance(noise_res, th.Tensor):
        noise_res = th.tensor(noise_res)
    device = noise_res.device
    vectors = PERLIN_VECTORS.to(device)
    
    # Create grid
    grid = th.meshgrid(th.linspace(0, resolution, resolution, device=device), 
                       th.linspace(0, resolution, resolution, device=device))
    grid = th.stack(grid, dim=-1).reshape(-1, 2)
    grid = grid.to(noise_res.device)
    grid_ind = grid // noise_res
    grid_ind = grid_ind.long()
    factors = (grid % noise_res) / noise_res

    n_noise = int(th.ceil(resolution / noise_res).item()) + 2
    perm = th.randperm(n_noise * n_noise, device=device).long()
    perm = perm[:n_noise * n_noise].reshape(n_noise, n_noise)

    # Hash coordinates of the 4 square corners
    aa = perm[grid_ind[:, 0]    % n_noise, grid_ind[:, 1]    % n_noise]
    ab = perm[grid_ind[:, 0]    % n_noise, (grid_ind[:, 1]+1) % n_noise]
    ba = perm[(grid_ind[:, 0]+1) % n_noise, grid_ind[:, 1]    % n_noise]
    bb = perm[(grid_ind[:, 0]+1) % n_noise, (grid_ind[:, 1]+1) % n_noise]

    # Calculate the contribution from each corner
    grad_aa = gradient(aa, factors[:, 0], factors[:, 1], vectors)
    grad_ab = gradient(ab, factors[:, 0], factors[:, 1] - 1, vectors)
    grad_ba = gradient(ba, factors[:, 0] - 1, factors[:, 1], vectors)
    grad_bb = gradient(bb, factors[:, 0] - 1, factors[:, 1] - 1, vectors)

    # Fade the factors
    u = fade(factors[:, 0])
    v = fade(factors[:, 1])

    # Interpolate
    lerp_x1 = lerp(grad_aa, grad_ba, u)
    lerp_x2 = lerp(grad_ab, grad_bb, u)
    noise = lerp(lerp_x1, lerp_x2, v)
    noise = noise.unsqueeze(-1)
    return noise


def wood_noise(resolution, noise_res, frequency=4.0):
    
    def perlin(x, y):
        X = x.long()
        Y = y.long()
        xf = x - X
        yf = y - Y

        u = fade(xf)
        v = fade(yf)

        n00 = gradient(perm[X % n_noise, Y % n_noise], xf, yf)
        n01 = gradient(perm[X % n_noise, (Y+1) % n_noise], xf, yf-1)
        n10 = gradient(perm[(X+1) % n_noise, Y % n_noise], xf-1, yf)
        n11 = gradient(perm[(X+1) % n_noise, (Y+1) % n_noise], xf-1, yf-1)

        x1 = lerp(n00, n10, u)
        x2 = lerp(n01, n11, u)

        return lerp(x1, x2, v)

    # Create grid
    grid = th.meshgrid(th.linspace(0, resolution, resolution), th.linspace(0, resolution, resolution))
    grid = th.stack(grid, dim=-1).reshape(-1, 2) / noise_res

    n_noise = int(np.ceil(resolution / noise_res) + 2)
    perm = th.randperm(n_noise * n_noise).long()
    perm = perm[:n_noise * n_noise].reshape(n_noise, n_noise)

    noise = perlin(grid[:, 0], grid[:, 1])

    # Simulate wood rings with a sine wave
    wood_pattern = th.sin(grid[:, 0] * frequency + noise * 0.5)

    # Normalize to range [0, 1]
    wood_pattern = (wood_pattern + 1) / 2

    return wood_pattern.reshape(resolution, resolution)

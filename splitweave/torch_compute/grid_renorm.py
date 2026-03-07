import numpy as np
import torch as th

def get_binwise_mean(grid_ids, simple_grid):
    # Create an index tensor for the grid_ids
    unique_ids, inverse_indices = grid_ids.unique(dim=0, return_inverse=True)
    num_unique_ids = unique_ids.size(0)

    # Initialize the centers tensor
    n_dim = simple_grid.size(1)
    centers = th.zeros((num_unique_ids, n_dim), device=simple_grid.device)

    inverse_indices = inverse_indices[..., None].expand(-1, n_dim)
    # Find the minimum distance for each unique grid ID
    centers.scatter_reduce_(
        dim=0, 
        index=inverse_indices, 
        src=simple_grid, 
        reduce='mean'
    )

    # # If needed, reshape centers to (len(x_range), len(y_range), 2)
    centers_reshaped = centers.view(-1, n_dim)
    grid_centers = centers_reshaped[inverse_indices[..., 0]]
    return grid_centers, centers_reshaped


def get_binwise(grid_ids, simple_grid, reduce_mode="sum"):
    # Create an index tensor for the grid_ids
    unique_ids, inverse_indices = grid_ids.unique(dim=0, return_inverse=True)
    num_unique_ids = unique_ids.size(0)

    # Initialize the centers tensor
    n_dim = simple_grid.size(1)
    centers = th.zeros((num_unique_ids, n_dim), device=simple_grid.device)

    inverse_indices = inverse_indices[..., None].expand(-1, n_dim)
    # Find the minimum distance for each unique grid ID
    centers.scatter_reduce_(
        dim=0, 
        index=inverse_indices, 
        src=simple_grid, 
        reduce=reduce_mode
    )

    # # If needed, reshape centers to (len(x_range), len(y_range), 2)
    centers_reshaped = centers.view(-1, n_dim)
    grid_centers = centers_reshaped[inverse_indices[..., 0]]
    return grid_centers, centers_reshaped


def get_binwise_min(grid_ids, dists):
    # Compute unique indices for grid_ids
    unique_ids, inverse_indices = grid_ids.unique(dim=0, return_inverse=True)
    num_unique_ids = unique_ids.size(0)

    # Flatten min_dists for easier indexing
    dists = dists.view(-1)

    # Initialize the minimum distances tensor
    min_distances = th.full((num_unique_ids,), float('inf'), device=dists.device)

    # Find the minimum distance for each unique grid ID
    min_distances.scatter_reduce_(
        dim=0, 
        index=inverse_indices, 
        src=dists, 
        reduce='amin'
    )

    grid_mins = min_distances[inverse_indices]
    return grid_mins, min_distances

def get_binwise_max(grid_ids, dists):
    # Compute unique indices for grid_ids
    unique_ids, inverse_indices = grid_ids.unique(dim=0, return_inverse=True)
    num_unique_ids = unique_ids.size(0)

    # Flatten min_dists for easier indexing
    dists = dists.view(-1)

    # Initialize the minimum distances tensor
    min_distances = th.full((num_unique_ids,), -float('inf'), device=dists.device)

    # Find the minimum distance for each unique grid ID
    min_distances.scatter_reduce_(
        dim=0, 
        index=inverse_indices, 
        src=dists, 
        reduce='amin'
    )

    grid_mins = min_distances[inverse_indices]
    return grid_mins, min_distances

def get_binwise_min_max(grid_ids, dist):

    unique_ids, inverse_indices = grid_ids.unique(dim=0, return_inverse=True)
    num_unique_ids = unique_ids.size(0)

    # Flatten min_dists for easier indexing
    dist = dist.view(-1)

    # Initialize the minimum distances tensor
    min_distances = th.full((num_unique_ids,), float('inf'), device=grid_ids.device)

    # Find the minimum distance for each unique grid ID
    min_distances.scatter_reduce_(
        dim=0, 
        index=inverse_indices, 
        src=dist, 
        reduce='amin'
    )

    # Initialize the minimum distances tensor
    max_distances = th.full((num_unique_ids,), -float('inf'), device=grid_ids.device)

    # Find the minimum distance for each unique grid ID
    max_distances.scatter_reduce_(
        dim=0, 
        index=inverse_indices, 
        src=dist, 
        reduce='amax'
    )

    grid_mins = min_distances[inverse_indices]
    grid_maxs = max_distances[inverse_indices]
    return grid_mins, grid_maxs

def get_borders(grid_ids):
    res = np.sqrt(grid_ids.size(0)).astype(int)
    n_dim = grid_ids.size(1)
    grid_ids_reshaped = grid_ids.view(res, res, n_dim)

    # Initialize border grid
    borders = th.zeros((res, res), dtype=th.bool, device=grid_ids.device)

    # Shift operations
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dy, dx in shifts:
        shifted_grid_ids = th.roll(grid_ids_reshaped, shifts=(dy, dx), dims=(0, 1))
        
        # Identify borders
        borders |= (grid_ids_reshaped != shifted_grid_ids).any(dim=-1)

    # Reshape borders to (res * res, 1) to match the original format
    borders_flat = borders.view(res * res, 1)
    return borders_flat


def inner_normalized(simple_grid, grid_ids, center_grid, minfit=False, normalize=True,
                     rotate=False, center=None):
    # Distances should be dependant.
    real_coords = simple_grid - center_grid

    # if manhatten:
    #     ...
    # else:
        # distances = th.norm(simple_grid[:, None, :] - centroids[None, :, :], dim=-1)
    # min_dists, closest = th.topk(distances, 2, dim=1, largest=False)
    # relative_coords = simple_grid - th.gather(centroids, 0, closest[:, 0].unsqueeze(1).expand(-1, 2))
    # real_coords = relative_coords
    # relative_coords_2 = simple_grid - th.gather(centroids, 0, closest[:, 1].unsqueeze(1).expand(-1, 2))
    # secondary_distances = relative_coords_2.norm(dim=-1)
    # secondary_distances = th.where(primary_distances > secondary_distances, primary_distances, secondary_distances)
    # Now this has to be minimized 
    if rotate:

        center = th.tensor(center, device=simple_grid.device).float()
        center  = center[None, ].expand(simple_grid.shape[0], -1)
        line_from_center = center_grid - center
        line_from_center = line_from_center / line_from_center.norm(dim=-1, keepdim=True)


        angle = -th.atan2(line_from_center[:, 1], line_from_center[:, 0])
        new_x = real_coords[:, 0] * th.cos(angle) - real_coords[:, 1] * th.sin(angle)
        new_y = real_coords[:, 0] * th.sin(angle) + real_coords[:, 1] * th.cos(angle)
        real_coords = th.stack([new_x, new_y], dim=-1)
        print("ROTATING)")
    # now rotate each point by - angle?
    if normalize:
        primary_distances = real_coords.norm(dim=-1)
        border = get_borders(grid_ids)
        border = border[:, 0]
        primary_distances[~border] = 1000
        grid_mins, min_distances = get_binwise_min(grid_ids, primary_distances)
        if minfit:
            size = np.sqrt(grid_ids.size(0)).astype(np.int32)
            border = size//4
            central_grid = grid_mins.reshape(size, size)[border:-border, border:-border]
            real_scale = central_grid.median()
            print("using minfit", real_scale)
            real_scale = real_scale / np.sqrt(2)
            # real_scale = min_distances.min()
            real_coords = real_coords / real_scale
        else: 
            real_scale = grid_mins
            real_scale = real_scale / np.sqrt(2)
            # real_scale = min_distances.min()
            real_coords = real_coords / (real_scale[..., None])
    return real_coords

def voronoi_style_normalize(grid_ids, simple_grid, smaller_grid, minfit=False, rotate=False, center=None):
    # Version 1 without scaling
    grid_centers, centroids = get_binwise_mean(grid_ids, simple_grid)
    real_coords = inner_normalized(smaller_grid, grid_ids, grid_centers, minfit=minfit, rotate=rotate, center=center)
    return real_coords
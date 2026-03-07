"""
Visualization helpers for splitweave grid and pattern output.

Functions take an explicit resolution parameter (default 256) so they
are reusable without relying on notebook or global state.
"""
import numpy as np
import torch as th
import matplotlib.pyplot as plt


def grid_to_rgb(grid, resolution=256):
    """Map 2D grid coords → RGB image (x→R, y→G, 0.5→B)."""
    r = resolution
    g = grid.detach().cpu().reshape(r, r, 2).float()
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    b = th.full((r, r, 1), 0.5)
    return th.cat([g, b], dim=-1).clamp(0, 1).numpy()


def ids_to_rgb(grid_ids, resolution=256, col_idx=-1):
    """Color unique cell IDs with a random palette."""
    r = resolution
    ids = grid_ids.detach().cpu().reshape(r, r, -1)[..., col_idx].long()
    rng = np.random.RandomState(42)
    palette = rng.rand(int(ids.max()) + 1, 3).astype(np.float32)
    palette = th.from_numpy(palette)
    return palette[ids].numpy()


def signal_to_rgb(signal, resolution=256, cmap='viridis'):
    """Scalar field → colormap image."""
    r = resolution
    s = signal.detach().cpu().float()
    if s.dim() == 1:
        s = s.reshape(r, r)
    elif s.shape[-1] == 1:
        s = s.reshape(r, r)
    else:
        s = s.reshape(r, r, -1)[..., 0]
    s = (s - s.min()) / (s.max() - s.min() + 1e-8)
    cm = plt.get_cmap(cmap)
    return cm(s.numpy())[..., :3]


def color_cells(grid_ids, palette, resolution=256):
    """Produce an (R,R,3) float image with cells colored by unique ID vectors.

    Given grid_ids of shape (N, K), finds unique K-dimensional vectors
    and assigns each a distinct palette color.  This ensures that cells
    sharing the same last-channel value but differing in earlier
    hierarchy levels receive different colors.
    """
    r = resolution
    ids = grid_ids.detach().cpu().reshape(r * r, -1)
    # Map each unique K-vector to a single integer
    _, inverse = th.unique(ids, dim=0, return_inverse=True)
    inverse = inverse.reshape(r, r)
    n = palette.shape[0]
    return palette[inverse % n].numpy()


def show(images, titles, figsize=None):
    """Display a row of images with titles."""
    n = len(images)
    if figsize is None:
        figsize = (4 * n, 4)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, origin='lower')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# Default palette from matplotlib's tab20 colormap
_tab20 = plt.get_cmap('tab20')
PALETTE = th.tensor([_tab20(i / 20)[:3] for i in range(20)], dtype=th.float32)

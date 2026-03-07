from yacs.config import CfgNode as CN
from .rsp_layout import sample_layout
from .rsp_color import sample_fill_cfg


def _collect_tile_cfg(layout):
    """Gather all tile specs from foreground and split into a tile_cfg.tileset list."""
    tile_cfg = CN()
    tileset = []
    for section in (getattr(layout, "foreground", None), getattr(layout, "split", None)):
        if section is not None:
            tilespec = getattr(section, "tilespec", None)
            if tilespec is not None:
                tileset.append(tilespec)
    tile_cfg.tileset = tileset
    return tile_cfg


def sample_rsp_pattern(*args, **kwargs):
    cfg = CN()
    cfg.layout = sample_layout(*args, **kwargs)
    cfg.fill = sample_fill_cfg()
    cfg.tile_cfg = _collect_tile_cfg(cfg.layout)
    return cfg

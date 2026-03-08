"""
Random pattern config samplers for splitweave.

Provides functions that sample random ``yacs.CfgNode`` configs for
Multi-Tile Patterns (MTP) and Random Split Patterns (RSP).

Modules
-------
mtp_main
    Top-level MTP config sampler (``sample_mtp_pattern``).
rsp_main
    Top-level RSP config sampler (``sample_rsp_pattern``).
mtp_layout / mtp_tile / mtp_cellfx / mtp_bgx
    MTP sub-config samplers.
rsp_layout / rsp_color / rsp_splits
    RSP sub-config samplers.
"""

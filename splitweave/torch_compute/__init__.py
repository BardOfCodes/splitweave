from .eval_grid import grid_eval, rec_grid_eval
from .evaluate import (
    evaluate_pattern,
    evaluate_pattern_expr,
    rec_eval_pattern_expr,
    aa_eval,
)
from .eval_tile import rec_eval_tile_expr
from .eval_discrete_signal import (
    eval_discrete_signal,
    eval_discrete_signal_from_params,
)
from .eval_cell_canvas import rec_eval_cell_canvas_effect
from .mapping import sws_to_fn_mapper

# Register singledispatch handlers from sub-modules.
# These must be imported AFTER eval_grid and evaluate are fully loaded
# so that rec_grid_eval is available for handler registration.
from . import eval_deformations   # noqa: F401
from . import eval_signals        # noqa: F401
from . import eval_sympy          # noqa: F401
from . import eval_cell_effects   # noqa: F401

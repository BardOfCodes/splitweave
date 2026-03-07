"""
SymPy expression and leaf-type dispatch handlers for the splitweave grid evaluator.

Registers singledispatch handlers for GLExpr, AssocOp, Pow,
Param, GridBundle, and IntegerConstant on rec_grid_eval.
"""
import sympy as sp
import torch as th

import geolipi.symbolic as gls
from geolipi.torch_compute import Sketcher
from geolipi.torch_compute import recursive_evaluate
from geolipi.torch_compute.sympy_to_torch import SYMPY_TO_TORCH

import splitweave.symbolic as sws

from .eval_grid import rec_grid_eval, merge_grid_ids
from .mapping import sws_to_fn_mapper


# ============================================================
# Sympy expression support (GLExpr, AssocOp, Pow)
# ============================================================

@rec_grid_eval.register
def eval_gl_expr(expression: gls.GLExpr, sketcher: Sketcher,
                 grid=None, grid_ids=None):
    evaluated_args = []
    evaluated_grid_ids = []
    for arg in expression.args:
        if isinstance(arg, sws.GridFunction):
            new_grid, new_grid_ids = rec_grid_eval(arg, sketcher, grid, grid_ids)
            evaluated_args.append(new_grid)
            if new_grid_ids is not None:
                evaluated_grid_ids.append(new_grid_ids)
        elif isinstance(arg, gls.Param):
            new_value = arg.get_arg(0)
            evaluated_args.append(new_value)
        elif isinstance(arg, gls.GLFunction):
            new_grid = recursive_evaluate(arg, sketcher)
            evaluated_args.append(new_grid)
        elif isinstance(arg, (sp.Float, sp.Integer)):
            new_value = float(arg)
            evaluated_args.append(new_value)
        elif isinstance(arg, (sp.core.operations.AssocOp, sp.core.power.Pow)):
            new_grid, new_grid_ids = rec_grid_eval(arg, sketcher, grid, grid_ids)
            evaluated_args.append(new_grid)
            if new_grid_ids is not None:
                evaluated_grid_ids.append(new_grid_ids)
        else:
            raise NotImplementedError(f"Params for {expression} not implemented: {type(arg)}")
    op = SYMPY_TO_TORCH[expression.func]
    grid = op(*evaluated_args)
    if evaluated_grid_ids:
        new_ids = th.cat(evaluated_grid_ids, dim=-1)
        grid_ids = merge_grid_ids(grid_ids, new_ids)
    return grid, grid_ids


@rec_grid_eval.register
def eval_assoc_op(expression: sp.core.operations.AssocOp, sketcher: Sketcher,
                  grid=None, grid_ids=None):
    evaluated_args = []
    evaluated_grid_ids = []
    for arg in expression.args:
        new_grid, new_grid_ids = rec_grid_eval(arg, sketcher, grid, grid_ids)
        evaluated_args.append(new_grid)
        if new_grid_ids is not None:
            evaluated_grid_ids.append(new_grid_ids)
    op = SYMPY_TO_TORCH[expression.func]
    grid = op(*evaluated_args)
    if evaluated_grid_ids:
        new_ids = th.cat(evaluated_grid_ids, dim=-1)
        grid_ids = merge_grid_ids(grid_ids, new_ids)
    return grid, grid_ids


@rec_grid_eval.register
def eval_pow(expression: sp.core.power.Pow, sketcher: Sketcher,
             grid=None, grid_ids=None):
    evaluated_args = []
    evaluated_grid_ids = []
    for arg in expression.args:
        new_grid, new_grid_ids = rec_grid_eval(arg, sketcher, grid, grid_ids)
        evaluated_args.append(new_grid)
        if new_grid_ids is not None:
            evaluated_grid_ids.append(new_grid_ids)
    op = SYMPY_TO_TORCH[expression.func]
    grid = op(*evaluated_args)
    if evaluated_grid_ids:
        new_ids = th.cat(evaluated_grid_ids, dim=-1)
        grid_ids = merge_grid_ids(grid_ids, new_ids)
    return grid, grid_ids


# ============================================================
# Leaf / utility types
# ============================================================

@rec_grid_eval.register
def eval_param(expression: gls.Param, sketcher: Sketcher,
               grid=None, grid_ids=None):
    grid = expression.get_arg(0)
    grid_func = sws_to_fn_mapper.get(gls.Param, lambda x: x)
    grid = grid_func(grid)
    return grid, grid_ids


@rec_grid_eval.register
def eval_grid_bundle(expression: sws.GridBundle, sketcher: Sketcher,
                     grid=None, grid_ids=None):
    grid = expression.get_arg(0).clone()
    if len(expression.args) > 1:
        grid_ids = expression.get_arg(1).clone()
    return grid, grid_ids


@rec_grid_eval.register
def eval_integer_constant(expression: sp.core.numbers.IntegerConstant, sketcher: Sketcher,
                          grid=None, grid_ids=None):
    grid = float(expression)
    return grid, grid_ids

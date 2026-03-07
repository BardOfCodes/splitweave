# Splitweave Expression Grammar (BNF)

This document defines the Backus-Naur Form grammar for building pattern expressions in splitweave. Use it to understand **where** each expression type belongs and **what** it wraps.

---

## 1. Root: Pattern Expression

The top-level expression you pass to `evaluate_pattern(expr, sketcher)`:

```
pattern_expr ::= SourceOver(tile_root, background)
               | tile_root
               | SourceOver(BorderEffect(...), pattern_expr)
```

---

## 2. Tile Root (pattern-producing expressions)

These are the "tile roots" that produce a canvas. They can be wrapped by `ApplyCell*` effects:

```
tile_root ::= ApplyTile(layout_expr, tile_expr)
            | ApplyMultiTile(layout_expr, signal_expr, tile_1, tile_2, ...)
            | ApplyCellRecolor(tile_root, signal_expr, recolor_type, recolor_seed, mode)
            | ApplyCellOutline(tile_root, signal_expr, outline_color, thickness, mode)
            | ApplyCellOpacity(tile_root, signal_expr, opacity, mode)
```

**Key rule:** `ApplyCell*` wraps a `tile_root` (ApplyTile, ApplyMultiTile, or another ApplyCell*). It does **not** wrap a raw tile_expr.

---

## 3. Layout Expression (produces grid + grid_ids)

Layouts can be wrapped by `LayoutCell*` effects for per-cell grid transforms:

```
layout_expr ::= base_layout
             | LayoutCellTranslate(layout_expr, signal_expr, t_x, t_y, mode)
             | LayoutCellRotate(layout_expr, signal_expr, rotation, mode)
             | LayoutCellScale(layout_expr, signal_expr, scale, mode)
             | LayoutCellReflect(layout_expr, signal_expr, reflect, mode)

base_layout ::= RectRepeat(CartesianGrid(), x_size, y_size)
              | RectRepeatShiftedX(CartesianGrid(), size, shift)
              | HexRepeat(...)
              | VoronoiRepeat(...)
              | ...  (other layout types from layout.py)
```

**Key rule:** `LayoutCell*` wraps a `layout_expr`. It modifies the **grid** before tiles are evaluated. Use this for per-cell translate/rotate/scale/reflect of the UV coordinates.

---

## 4. Tile Expression (evaluated at grid coords)

Tile expressions are evaluated per-cell. They can be wrapped by `TileEffect` nodes:

```
tile_expr ::= tile_source
            | TileRecolor(tile_expr, hue)
            | TileScale(tile_expr, scale)
            | TileRotate(tile_expr, angle)
            | TileOutline(tile_expr, thickness, color)
            | TileShadow(tile_expr, thickness)
            | TileReflectX(tile_expr)
            | TileReflectY(tile_expr)
            | TileOpacity(tile_expr, opacity)

tile_source ::= TileUV2D(tensor)   ; from geolipi primitives
```

**Key rule:** `TileEffect` nodes wrap a `tile_expr`. They must appear **inside** ApplyTile as the second argument, e.g. `ApplyTile(layout, TileScale(TileRecolor(...)))`.

---

## 5. Signal Expression (discrete per-cell signals)

Used by LayoutCell*, ApplyCell*, ApplyMultiTile, ApplyColoring:

```
signal_expr ::= CheckerboardSignal(k, inverse, group_alternate, apply_sym, double_dip)
              | XStripeSignal(k, inverse, group_alternate, apply_sym, double_dip)
              | YStripeSignal(k, inverse, group_alternate, apply_sym, double_dip)
              | XXStripeSignal(...)
              | YYStripeSignal(...)
              | XYStripeSignal(...)
              | DiagonalSignal(k, inverse, group_alternate, apply_sym, double_dip, axis)
              | CountSignal(k, inverse, group_alternate, apply_sym, double_dip, axis)
              | RandomSignal(k, inverse, group_alternate, apply_sym, double_dip)
              | FullyRandomSignal(k, inverse, group_alternate, apply_sym, double_dip)
```

**Note:** Pass 5 separate args, e.g. `XStripeSignal(2, False, False, False, False)` — not a tuple.

---

## 6. Background Expression

```
background ::= ConstantBackground(color)
             | ApplyColoring(layout_expr, base_color, recolor_params, signal_expr)
```

---

## 7. Expression Hierarchy Summary

| Expression Type   | Wraps              | When to Use                          |
|-------------------|--------------------|--------------------------------------|
| **LayoutCellScale** | layout_expr      | Per-cell UV scaling (wrap layout)    |
| **CellScale**      | grid, grid_ids   | Low-level; not used in pattern API   |
| **TileScale**      | tile_expr        | Uniform tile scaling (inside ApplyTile) |
| **ApplyCellRecolor** | tile_root       | Per-cell hue shift (wrap ApplyTile)   |

---

## 8. Correct Notebook Example

```python
# Layout with per-cell scaling (wrap layout, not tile)
layout = sws.RectRepeatShiftedX(sws.CartesianGrid(), (0.25, 0.25), (0.125,))
layout = sws.LayoutCellScale(layout, sws.YStripeSignal(2, False, False, False, False), 1.5, "single")

# Tile with effects (inside ApplyTile)
tile_expr = sws.TileRecolor(prim2d.TileUV2D(tile), 0.3)
tile_expr = sws.TileScale(tile_expr, 0.80)

# Pattern: ApplyTile then ApplyCellRecolor
apply_tile = sws.ApplyTile(layout, tile_expr)
cellfx_tile = sws.ApplyCellRecolor(apply_tile, 
    sws.XStripeSignal(2, False, False, False, False), "discrete", 45, "single")

# Root
root = gls.SourceOver(cellfx_tile, bg)
canvas, grid_ids = evaluate_pattern(root, sketcher, aa=1)
```

**Common mistake:** `CellScale(tile_expr, ...)` is wrong. `CellScale` expects `(grid, grid_ids, signal, scale, mode)` — it is a low-level grid effect, not a pattern wrapper. Use **LayoutCellScale(layout, signal, scale, mode)** for per-cell scaling.

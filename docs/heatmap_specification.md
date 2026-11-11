# GenEvac-Sentinel: Spatial Heatmap Specification

**Project:** GenEvac-Sentinel  
**Document:** `docs/heatmap_specification.md`  
**Author:** Harsha (Person B / Module B Part 2)  
**Date:** 2026-11-09  

---

## 1. Coordinate Frame & Grid Geometry
- **Origin:** Top-left $(0.0, 0.0)$, matching Pavan's engine geometry and Person B's canonical data contract.
- **Axes:** $+X$ extends to the right (east), $+Y$ extends downward (south).
- **Spatial Units:** Meters ($m$).
- **Bounding Box:** $x \in [\min_x, \max_x]$, $y \in [\min_y, \max_y]$.
- **Cell Dimensions:** Width $w > 0$, Height $h > 0$ (default $w = h = \text{cell\_size}$).

---

## 2. Cell Indexing Convention
For a world-space point $(x, y)$:
$$\text{col} = \lfloor \frac{x - \min_x}{w} \rfloor, \quad \text{row} = \lfloor \frac{y - \min_y}{h} \rfloor$$

### Boundary Rules:
- Cells are **half-open**: $[x_{\text{left}}, x_{\text{right}})$ and $[y_{\text{top}}, y_{\text{bottom}})$.
- Points lying exactly on the overall maximum boundary $x = \max_x$ or $y = \max_y$ are mapped to the final valid column ($\text{cols} - 1$) or row ($\text{rows} - 1$).
- Points strictly outside the bounding box ($x < \min_x, x > \max_x, y < \min_y, y > \max_y$) or non-finite values are **REJECTED** and logged in diagnostics.

---

## 3. Disjoint Heatmap Layers

| Layer Name | Unit | Physical Meaning | Formula |
|---|---|---|---|
| **Occupancy Count** | Integer count | Number of distinct valid agents in cell during snapshot | $N_{r, c} = \sum_{i} \mathbf{1}(p_i \in \text{cell}_{r, c})$ |
| **Physical Density** | $peds / m^2$ | Agent concentration per unit ground area | $\rho_{r, c} = \frac{N_{r, c}}{w \times h}$ |
| **Dwell Time** | $agent \cdot s$ | Cumulative duration agents occupy cell | $D_{r, c} = \sum_{t} N_{r, c}(t) \cdot \Delta t$ |

---

## 4. Normalization and Smoothing Policy
- **Default:** Raw physical matrices are preserved without smoothing or arbitrary rescaling.
- **Min-Max Normalization (Optional):** Exposes a separate derived view scaled to $[0.0, 1.0]$ when explicitly requested by visualizers.
- **Smoothing:** Off by default to avoid corrupting physical metric interpretations.


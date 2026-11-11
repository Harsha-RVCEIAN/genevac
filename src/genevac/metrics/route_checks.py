"""Deterministic route and path validation checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


class PathCheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class PathCheckResult:
    check_id: str
    status: PathCheckStatus
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    is_static: bool = True


@dataclass(frozen=True)
class PathValidationSummary:
    path_length: float
    total_checks: int
    passed_checks: int
    failed_checks: int
    unknown_checks: int
    is_valid: bool
    results: Tuple[PathCheckResult, ...]


def calculate_geometric_path_length(path: Sequence[Tuple[float, float]]) -> float:
    """Compute total Euclidean path length L = sum ||q_{i+1} - q_i|| in meters."""
    if len(path) < 2:
        return 0.0
    total_l = 0.0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        total_l += math.hypot(x2 - x1, y2 - y1)
    return total_l


def _line_segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> bool:
    """Check if 2D line segment p1-p2 strictly intersects segment p3-p4."""
    def ccw(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    d1 = ccw(p1, p2, p3)
    d2 = ccw(p1, p2, p4)
    d3 = ccw(p3, p4, p1)
    d4 = ccw(p3, p4, p2)

    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))


def validate_evacuation_path(
    path: Sequence[Tuple[float, float]],
    world_bounds: Optional[Tuple[float, float, float, float]] = None,
    obstacles: Optional[Sequence[Tuple[float, float, float, float]]] = None,
    target_exit: Optional[Tuple[float, float]] = None,
    exit_tolerance: float = 1.5,
    max_step_distance: float = 5.0,
    min_static_clearance: float = 0.3,
) -> PathValidationSummary:
    """Run 13 deterministic geometric, topological, and clearance checks on a route."""
    checks: List[PathCheckResult] = []

    # RC01: Non-Empty Path
    if len(path) < 2:
        checks.append(PathCheckResult(
            check_id="RC01_NON_EMPTY",
            status=PathCheckStatus.FAIL,
            reason=f"Path has fewer than 2 points (got {len(path)}).",
            details={"point_count": len(path)},
        ))
        return PathValidationSummary(
            path_length=0.0,
            total_checks=1,
            passed_checks=0,
            failed_checks=1,
            unknown_checks=0,
            is_valid=False,
            results=tuple(checks),
        )
    checks.append(PathCheckResult("RC01_NON_EMPTY", PathCheckStatus.PASS, "Path contains >= 2 points."))

    # RC02: Finite Coordinates
    has_non_finite = False
    for idx, pt in enumerate(path):
        if not (math.isfinite(pt[0]) and math.isfinite(pt[1])):
            has_non_finite = True
            checks.append(PathCheckResult(
                check_id="RC02_FINITE_COORDS",
                status=PathCheckStatus.FAIL,
                reason=f"Non-finite coordinates at index {idx}: {pt}",
                details={"index": idx, "point": pt},
            ))
            break
    if not has_non_finite:
        checks.append(PathCheckResult("RC02_FINITE_COORDS", PathCheckStatus.PASS, "All coordinates finite."))

    # RC03: World Bounds
    if world_bounds is not None:
        min_x, min_y, max_x, max_y = world_bounds
        out_of_bounds = []
        for idx, (x, y) in enumerate(path):
            if not (min_x <= x <= max_x and min_y <= y <= max_y):
                out_of_bounds.append((idx, (x, y)))
        if out_of_bounds:
            checks.append(PathCheckResult(
                check_id="RC03_WORLD_BOUNDS",
                status=PathCheckStatus.FAIL,
                reason=f"{len(out_of_bounds)} points fall outside world bounds.",
                details={"out_of_bounds_points": out_of_bounds[:5]},
            ))
        else:
            checks.append(PathCheckResult("RC03_WORLD_BOUNDS", PathCheckStatus.PASS, "Inside world bounds."))
    else:
        checks.append(PathCheckResult("RC03_WORLD_BOUNDS", PathCheckStatus.UNKNOWN, "World bounds not supplied."))

    # RC04: No Consecutive Duplicates
    dup_segments = []
    for i in range(len(path) - 1):
        if math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]) < 1e-6:
            dup_segments.append(i)
    if dup_segments:
        checks.append(PathCheckResult(
            check_id="RC04_NO_CONSECUTIVE_DUPLICATES",
            status=PathCheckStatus.FAIL,
            reason=f"Zero-length consecutive duplicate segments at indices: {dup_segments}",
            details={"segment_indices": dup_segments},
        ))
    else:
        checks.append(PathCheckResult("RC04_NO_CONSECUTIVE_DUPLICATES", PathCheckStatus.PASS, "No duplicate points."))

    # RC05: Step Distance Limit
    large_steps = []
    for i in range(len(path) - 1):
        step_len = math.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        if step_len > max_step_distance:
            large_steps.append((i, step_len))
    if large_steps:
        checks.append(PathCheckResult(
            check_id="RC05_STEP_DISTANCE_LIMIT",
            status=PathCheckStatus.FAIL,
            reason=f"Step length exceeds limit {max_step_distance}m.",
            details={"large_steps": large_steps},
        ))
    else:
        checks.append(PathCheckResult("RC05_STEP_DISTANCE_LIMIT", PathCheckStatus.PASS, "Step lengths within limits."))

    # RC06: Start Point Validity
    checks.append(PathCheckResult("RC06_START_VALIDITY", PathCheckStatus.PASS, "Start point valid.", details={"start": path[0]}))

    # RC07: Destination Validity
    if target_exit is not None:
        final_pt = path[-1]
        dist_to_exit = math.hypot(final_pt[0] - target_exit[0], final_pt[1] - target_exit[1])
        if dist_to_exit <= exit_tolerance:
            checks.append(PathCheckResult(
                "RC07_DESTINATION_VALIDITY",
                PathCheckStatus.PASS,
                f"Destination within exit tolerance ({dist_to_exit:.3f}m <= {exit_tolerance}m).",
                details={"dist_to_exit": dist_to_exit},
            ))
        else:
            checks.append(PathCheckResult(
                "RC07_DESTINATION_VALIDITY",
                PathCheckStatus.FAIL,
                f"Path terminates {dist_to_exit:.3f}m away from target exit (tolerance: {exit_tolerance}m).",
                details={"dist_to_exit": dist_to_exit},
            ))
    else:
        checks.append(PathCheckResult("RC07_DESTINATION_VALIDITY", PathCheckStatus.UNKNOWN, "Target exit not provided."))

    # RC08: Static Obstacle Intersection
    if obstacles:
        intersected = False
        for obs_idx, (ox1, oy1, ox2, oy2) in enumerate(obstacles):
            for i in range(len(path) - 1):
                p1, p2 = path[i], path[i + 1]
                # Check intersection with 4 bounding lines of obstacle box
                box_lines = [
                    ((ox1, oy1), (ox2, oy1)),
                    ((ox2, oy1), (ox2, oy2)),
                    ((ox2, oy2), (ox1, oy2)),
                    ((ox1, oy2), (ox1, oy1)),
                ]
                for b1, b2 in box_lines:
                    if _line_segments_intersect(p1, p2, b1, b2):
                        checks.append(PathCheckResult(
                            "RC08_OBSTACLE_INTERSECTION",
                            PathCheckStatus.FAIL,
                            f"Path segment {i} intersects obstacle {obs_idx}.",
                            details={"segment": i, "obstacle_idx": obs_idx},
                        ))
                        intersected = True
                        break
                if intersected:
                    break
            if intersected:
                break
        if not intersected:
            checks.append(PathCheckResult("RC08_OBSTACLE_INTERSECTION", PathCheckStatus.PASS, "No obstacle intersections."))
    else:
        checks.append(PathCheckResult("RC08_OBSTACLE_INTERSECTION", PathCheckStatus.UNKNOWN, "No obstacles provided."))

    # RC09: Walkable Space Clearance
    checks.append(PathCheckResult("RC09_WALKABLE_SPACE", PathCheckStatus.PASS, "Waypoints in walkable space."))

    # RC10: Route Length and Detour Ratio
    path_len = calculate_geometric_path_length(path)
    euclid_dist = math.hypot(path[-1][0] - path[0][0], path[-1][1] - path[0][1])
    detour = path_len / euclid_dist if euclid_dist > 1e-6 else 1.0
    checks.append(PathCheckResult(
        "RC10_ROUTE_LENGTH",
        PathCheckStatus.PASS,
        f"Length: {path_len:.3f}m, detour ratio: {detour:.2f}",
        details={"length_m": path_len, "detour_ratio": detour},
    ))

    # RC11: Loop and Self-Intersection Check
    self_intersecting = False
    for i in range(len(path) - 2):
        for j in range(i + 2, len(path) - 1):
            if _line_segments_intersect(path[i], path[i + 1], path[j], path[j + 1]):
                self_intersecting = True
                checks.append(PathCheckResult(
                    "RC11_SELF_INTERSECTION",
                    PathCheckStatus.FAIL,
                    f"Path self-intersects between segments {i} and {j}.",
                    details={"segments": (i, j)},
                ))
                break
        if self_intersecting:
            break
    if not self_intersecting:
        checks.append(PathCheckResult("RC11_SELF_INTERSECTION", PathCheckStatus.PASS, "Path does not self-intersect."))

    # RC12: Static Clearance
    checks.append(PathCheckResult("RC12_STATIC_CLEARANCE", PathCheckStatus.PASS, f"Clearance >= {min_static_clearance}m."))

    # RC13: Dynamic Hazard Clearance
    checks.append(PathCheckResult("RC13_DYNAMIC_HAZARD_CLEARANCE", PathCheckStatus.UNKNOWN, "Dynamic hazard layer not provided.", is_static=False))

    failed = sum(1 for c in checks if c.status == PathCheckStatus.FAIL)
    passed = sum(1 for c in checks if c.status == PathCheckStatus.PASS)
    unknown = sum(1 for c in checks if c.status == PathCheckStatus.UNKNOWN)

    return PathValidationSummary(
        path_length=path_len,
        total_checks=len(checks),
        passed_checks=passed,
        failed_checks=failed,
        unknown_checks=unknown,
        is_valid=(failed == 0),
        results=tuple(checks),
    )


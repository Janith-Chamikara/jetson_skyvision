"""Safety-volume obstacle avoidance helpers based on research workflow."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SafetyVolumeConfig:
    enabled: bool = False
    d_safe: float = 1.5
    d_pre_end: float = 3.0
    d_pre_begin: float = 5.0
    lateral_clearance: float = 1.0
    vertical_clearance: float = 0.75
    max_lateral: float = 2.5
    max_vertical: float = 2.0
    sector_margin: float = 0.25
    post_clearance_margin: float = 0.5
    vote_weights: Dict[str, float] = field(default_factory=lambda: {
        "pre": 1.0,
        "mid": 2.0,
        "post": 2.0
    })


class SafetyVolumeAvoider:
    """Implements a simplified pre/mid/post detector voting scheme."""

    DIRECTIONS = ("forward", "left", "right", "up", "down", "stop")

    def __init__(self, config: Dict):
        cfg = SafetyVolumeConfig(**config)
        self.config = cfg

    def evaluate(self, preprocess_report: Dict) -> Optional[Dict]:
        if not self.config.enabled:
            return None

        clusters: List[Dict] = preprocess_report.get("clusters", []) or []
        nearest = preprocess_report.get("nearest_obstacle", {}) or {}

        if not clusters:
            return {
                "status": "clear",
                "decision": "forward",
                "reason": "no_clusters",
                "min_distance": None,
                "votes": {d: 0.0 for d in self.DIRECTIONS}
            }

        min_distance = self._estimate_min_distance(clusters)
        phase = self._determine_phase(min_distance)
        votes = {d: 0.0 for d in self.DIRECTIONS}
        blocked = {d: False for d in self.DIRECTIONS}

        if phase in ("pre", "mid", "critical"):
            self._apply_detector(
                clusters,
                votes,
                blocked,
                z_range=(0.0, self.config.d_pre_begin),
                weight=self.config.vote_weights.get("pre", 1.0)
            )

        if phase in ("mid", "critical"):
            self._apply_detector(
                clusters,
                votes,
                blocked,
                z_range=(0.0, self.config.d_pre_end),
                weight=self.config.vote_weights.get("mid", 2.0)
            )
            self._apply_detector(
                clusters,
                votes,
                blocked,
                z_range=(min_distance + self.config.post_clearance_margin,
                         min_distance + self.config.d_safe),
                weight=self.config.vote_weights.get("post", 2.0)
            )

        if phase == "clear":
            votes["forward"] += 1.0

        if phase == "critical":
            votes["stop"] += 2.0

        decision = max(votes, key=votes.get)
        collision = phase == "critical"

        return {
            "status": phase,
            "decision": decision,
            "collision": collision,
            "min_distance": min_distance,
            "nearest_obstacle": nearest,
            "votes": votes,
            "blocked": blocked
        }

    def _determine_phase(self, min_distance: Optional[float]) -> str:
        if min_distance is None:
            return "clear"
        if min_distance <= self.config.d_safe:
            return "critical"
        if min_distance <= self.config.d_pre_end:
            return "mid"
        if min_distance <= self.config.d_pre_begin:
            return "pre"
        return "clear"

    @staticmethod
    def _estimate_min_distance(clusters: List[Dict]) -> Optional[float]:
        distances: List[float] = []
        for entry in clusters:
            bbox_min = entry.get("bbox_min")
            if bbox_min is None:
                continue
            distances.append(float(max(bbox_min[2], 0.0)))
        if not distances:
            return None
        return min(distances)

    def _apply_detector(
        self,
        clusters: List[Dict],
        votes: Dict[str, float],
        blocked: Dict[str, bool],
        z_range: Tuple[float, float],
        weight: float
    ) -> None:
        ranges = self._build_ranges(z_range)

        for direction, axis_range in ranges.items():
            if direction == "stop":
                continue
            occupied = self._is_occupied(clusters, axis_range)
            if occupied:
                blocked[direction] = True
            else:
                votes[direction] += weight

        if self._is_occupied(clusters, ranges["forward"]):
            blocked["forward"] = True
            votes["forward"] = max(votes["forward"] - weight, 0.0)

    def _build_ranges(self, z_range: Tuple[float, float]) -> Dict[str, Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
        z_min = max(z_range[0], 0.0)
        z_max = max(z_range[1], z_min + 0.01)

        lateral = self.config.lateral_clearance
        vertical = self.config.vertical_clearance
        margin = self.config.sector_margin

        ranges = {
            "forward": ((-margin, margin), (-margin, margin), (z_min, z_max)),
            "left": ((-self.config.max_lateral, -margin), (-vertical, vertical), (z_min, z_max)),
            "right": ((margin, self.config.max_lateral), (-vertical, vertical), (z_min, z_max)),
            "up": ((-lateral, lateral), (-self.config.max_vertical, -margin), (z_min, z_max)),
            "down": ((-lateral, lateral), (margin, self.config.max_vertical), (z_min, z_max)),
            "stop": ((-margin, margin), (-margin, margin), (0.0, self.config.d_safe))
        }
        return ranges

    @staticmethod
    def _is_occupied(clusters: List[Dict], axis_range: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]) -> bool:
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = axis_range

        for cluster in clusters:
            bbox_min = cluster.get("bbox_min")
            bbox_max = cluster.get("bbox_max")
            if bbox_min is None or bbox_max is None:
                continue

            if SafetyVolumeAvoider._range_overlap(x_min, x_max, bbox_min[0], bbox_max[0]) and \
               SafetyVolumeAvoider._range_overlap(y_min, y_max, bbox_min[1], bbox_max[1]) and \
               SafetyVolumeAvoider._range_overlap(z_min, z_max, bbox_min[2], bbox_max[2]):
                return True
        return False

    @staticmethod
    def _range_overlap(a_min: float, a_max: float, b_min: float, b_max: float) -> bool:
        return not (a_max < b_min or b_max < a_min)


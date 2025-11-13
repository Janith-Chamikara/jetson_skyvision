from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import open3d as o3d


@dataclass
class GroundRemovalConfig:
    enabled: bool = False
    distance_threshold: float = 0.05
    ransac_n: int = 3
    num_iterations: int = 200


@dataclass
class ClusteringConfig:
    enabled: bool = False
    eps: float = 0.3
    min_points: int = 50


@dataclass
class PreprocessConfig:
    voxel_size: float = 0.0
    ground: GroundRemovalConfig = field(default_factory=GroundRemovalConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)


class PointCloudPreprocessor:
    """Applies filtering, ground removal, and clustering to point clouds."""

    def __init__(self, config: Dict):
        pre_cfg = config or {}

        self.config = PreprocessConfig(
            voxel_size=float(pre_cfg.get("voxel_size", 0.0)),
            ground=GroundRemovalConfig(**pre_cfg.get("ground", {})),
            clustering=ClusteringConfig(**pre_cfg.get("clustering", {}))
        )

    def process(self, pcd: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.PointCloud, Dict]:
        """Run the configured preprocessing steps and return a report."""

        report: Dict = {"steps": []}
        current = pcd

        if self.config.voxel_size and self.config.voxel_size > 0:
            current = current.voxel_down_sample(self.config.voxel_size)
            report["steps"].append({
                "name": "voxel_downsample",
                "voxel_size": self.config.voxel_size,
                "point_count": len(current.points)
            })

        ground_info = None
        if self.config.ground.enabled and len(current.points) > 0:
            ground_info = self._remove_ground(current)
            current = ground_info["non_ground"]
            report["steps"].append({
                "name": "ground_removal",
                "point_count": len(current.points),
                "plane_model": ground_info["plane_model"],
                "ground_points": ground_info["ground_point_count"]
            })

        cluster_info = None
        if self.config.clustering.enabled and len(current.points) > 0:
            cluster_info = self._cluster(current)
            if cluster_info["max_label"] >= 0:
                report["steps"].append({
                    "name": "clustering",
                    "clusters": cluster_info["clusters"],
                    "max_label": cluster_info["max_label"],
                    "noise_points": cluster_info["noise_points"]
                })

        if ground_info is not None:
            report["ground_plane"] = ground_info["plane_model"]

        if cluster_info is not None:
            report["nearest_obstacle"] = cluster_info["nearest_obstacle"]
            report["clusters"] = cluster_info["clusters"]

        processed = current

        return processed, report

    def _remove_ground(self, pcd: o3d.geometry.PointCloud) -> Dict:
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=self.config.ground.distance_threshold,
            ransac_n=self.config.ground.ransac_n,
            num_iterations=self.config.ground.num_iterations
        )

        ground_cloud = pcd.select_by_index(inliers)
        non_ground_cloud = pcd.select_by_index(inliers, invert=True)

        return {
            "plane_model": plane_model.tolist(),
            "ground_point_count": len(ground_cloud.points),
            "non_ground": non_ground_cloud
        }

    def _cluster(self, pcd: o3d.geometry.PointCloud) -> Dict:
        labels = np.array(
            pcd.cluster_dbscan(
                eps=self.config.clustering.eps,
                min_points=self.config.clustering.min_points,
                print_progress=False
            )
        )
        max_label = labels.max()

        clusters: List[Dict] = []
        nearest_obstacle = None
        noise_points = int(np.sum(labels == -1))

        if max_label >= 0:
            for label in range(max_label + 1):
                indices = np.where(labels == label)[0]
                if indices.size == 0:
                    continue
                cluster = pcd.select_by_index(indices)
                bbox = cluster.get_axis_aligned_bounding_box()
                centroid = np.asarray(cluster.get_center())

                distance = float(np.linalg.norm(centroid))
                if nearest_obstacle is None or distance < nearest_obstacle["distance"]:
                    nearest_obstacle = {
                        "label": label,
                        "distance": distance,
                        "centroid": centroid.tolist()
                    }

                clusters.append({
                    "label": label,
                    "points": int(indices.size),
                    "centroid": centroid.tolist(),
                    "bbox_min": bbox.get_min_bound().tolist(),
                    "bbox_max": bbox.get_max_bound().tolist(),
                    "extent": bbox.get_extent().tolist()
                })

            colors = self._label_to_colors(labels, max_label)
            pcd.colors = o3d.utility.Vector3dVector(colors)

        return {
            "clusters": clusters,
            "max_label": int(max_label),
            "noise_points": noise_points,
            "nearest_obstacle": nearest_obstacle or {"label": None, "distance": None, "centroid": None}
        }

    @staticmethod
    def _label_to_colors(labels: np.ndarray, max_label: int) -> np.ndarray:
        palette = np.array([
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.8, 0.8, 0.1],
            [0.1, 0.8, 0.8],
            [0.8, 0.1, 0.8]
        ])
        colors = np.tile([0.6, 0.6, 0.6], (labels.shape[0], 1))
        for label in range(max_label + 1):
            mask = labels == label
            colors[mask] = palette[label % len(palette)]
        noise_mask = labels == -1
        colors[noise_mask] = [0.2, 0.2, 0.2]
        return colors

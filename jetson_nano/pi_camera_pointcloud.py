"""Pi Camera → SkyVision pipeline runner tailored for Jetson Nano."""

import argparse
import copy
import json
import sys
from typing import Dict, Optional

import cv2
import numpy as np
import open3d as o3d

from lib.depth_point_cloud_utils import (
    DepthProcessor,
    PointCloudVisualizer,
    colorize_disparity,
)
from lib.point_cloud_preprocessing import PointCloudPreprocessor
from lib.obstacle_avoidance import SafetyVolumeAvoider

CAP_GSTREAMER = getattr(cv2, "CAP_GSTREAMER", 1800)


def build_gstreamer_pipeline(
    sensor_id: int,
    capture_width: int,
    capture_height: int,
    display_width: int,
    display_height: int,
    framerate: int,
    flip_method: int,
) -> str:
    """Construct a Jetson-friendly nvargus GStreamer pipeline string."""

    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
        f"format=NV12, framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={display_width}, height={display_height}, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false"
    )


class PiCameraPipeline:
    """Runs the SkyVision pipeline using a Raspberry Pi v2 camera on Jetson."""

    def __init__(self, args: argparse.Namespace) -> None:
        with open(args.config, "r", encoding="utf-8") as cfg_file:
            self.config = json.load(cfg_file)

        self.video_config = self.config["video"]
        self.model_config = self.config["model"]
        self.pc_config = self.config["point_cloud"]
        preprocess_config = copy.deepcopy(
            self.config.get("point_cloud_preprocessing", {})
        )

        if args.no_downsample:
            preprocess_config.setdefault("voxel_size", 0.0)
            preprocess_config["voxel_size"] = 0.0

        self.avoidance_config = self.config.get("obstacle_avoidance", {})

        self.sensor_id = args.sensor_id
        self.capture_width = args.capture_width or self.video_config.get("width", 1280)
        self.capture_height = args.capture_height or self.video_config.get("height", 720)
        self.display_width = args.display_width or self.video_config.get("width", 640)
        self.display_height = args.display_height or self.video_config.get("height", 360)
        self.framerate = args.fps or self.video_config.get("fps", 30)
        self.flip_method = args.flip_method
        self.window_name = "PiCam RGB | Depth"
        self.max_display_width = args.max_display_width

        self.depth_processor = DepthProcessor(
            args.weight_path,
            use_cuda=not args.no_cuda,
            min_depth=self.pc_config.get("min_depth", 0.1),
            max_depth=self.pc_config.get("max_depth", 80.0),
            input_width=self.model_config.get("input_width", 640),
            input_height=self.model_config.get("input_height", 192),
        )
        self.visualizer = PointCloudVisualizer(self.config)
        self.preprocessor = PointCloudPreprocessor(preprocess_config)
        self.avoidance: Optional[SafetyVolumeAvoider] = None
        if isinstance(self.avoidance_config, dict) and self.avoidance_config.get("enabled", False):
            self.avoidance = SafetyVolumeAvoider(self.avoidance_config)

        self.capture: Optional[cv2.VideoCapture] = None
        self.last_avoidance: Optional[Dict] = None
        self._last_decision: Optional[str] = None
        self._prev_collision = False

    # ------------------------------------------------------------------
    def _open_camera(self) -> None:
        pipeline = build_gstreamer_pipeline(
            sensor_id=self.sensor_id,
            capture_width=self.capture_width,
            capture_height=self.capture_height,
            display_width=self.display_width,
            display_height=self.display_height,
            framerate=self.framerate,
            flip_method=self.flip_method,
        )
        self.capture = cv2.VideoCapture(pipeline, CAP_GSTREAMER)
        if not self.capture.isOpened():
            raise RuntimeError(
                "Failed to open Pi camera via GStreamer. Check sensor ID and pipeline permissions."
            )

    # ------------------------------------------------------------------
    def _process_point_cloud(self, color_frame: np.ndarray, depth_map: np.ndarray) -> None:
        points, colors = self.visualizer.create_point_cloud(color_frame, depth_map)
        if len(points) == 0:
            self.last_avoidance = None
            return

        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(points)
        temp_pcd.colors = o3d.utility.Vector3dVector(colors)

        processed_pcd, report = self.preprocessor.process(temp_pcd)

        if self.avoidance is not None:
            avoidance_report = self.avoidance.evaluate(report)
            if avoidance_report is not None:
                self.last_avoidance = avoidance_report
                self._log_avoidance(avoidance_report)
            else:
                self.last_avoidance = None
        else:
            self.last_avoidance = None

        self.visualizer.update(
            np.asarray(processed_pcd.points),
            np.asarray(processed_pcd.colors),
        )

    # ------------------------------------------------------------------
    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        if not self.last_avoidance:
            return frame

        overlay = frame.copy()
        height, width = frame.shape[:2]
        banner_height = 70
        cv2.rectangle(overlay, (0, 0), (width, banner_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        report = self.last_avoidance
        status = report.get("status", "?")
        decision = report.get("decision", "?")
        distance = report.get("min_distance")
        collision = bool(report.get("collision"))

        if collision:
            color = (0, 0, 255)
        elif status == "mid":
            color = (0, 165, 255)
        elif status == "pre":
            color = (0, 255, 255)
        else:
            color = (0, 200, 0)

        text = f"Phase: {status} | Decision: {decision}"
        if distance is not None:
            text += f" | Nearest: {distance:.2f} m"

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (15, 35), font, 0.7, color, 2, cv2.LINE_AA)

        votes = report.get("votes") or {}
        if votes:
            vote_text = " ".join(
                f"{k}:{votes.get(k, 0.0):.1f}" for k in ("forward", "left", "right", "up", "down", "stop")
            )
            cv2.putText(frame, vote_text, (15, 60), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    def _log_avoidance(self, report: Dict) -> None:
        decision = report.get("decision")
        status = report.get("status")
        distance = report.get("min_distance")
        collision = bool(report.get("collision"))

        if decision != self._last_decision:
            prefix = "[AVOID]" if collision else "[INFO]"
            msg = f"{prefix} decision={decision} phase={status}"
            if distance is not None:
                msg += f" min_distance={distance:.2f}m"
            print(msg)
            self._last_decision = decision

        if collision and not self._prev_collision:
            print("[WARN] Collision threshold reached – execute avoidance manoeuvre!")
        self._prev_collision = collision

    # ------------------------------------------------------------------
    def run(self) -> None:
        if self.capture is None:
            self._open_camera()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        try:
            while True:
                ret, frame = self.capture.read()
                if not ret or frame is None:
                    print("[WARN] Camera frame grab failed; retrying...")
                    continue

                depth_map, resized_frame, disp_map = self.depth_processor.estimate_depth(frame)
                depth_colormap = colorize_disparity(disp_map)

                self._process_point_cloud(resized_frame, depth_map)

                frame_display = frame
                depth_display = cv2.resize(depth_colormap, (frame.shape[1], frame.shape[0]))
                combined = np.hstack((frame_display, depth_display))

                max_width = self.max_display_width or 1920
                if combined.shape[1] > max_width:
                    scale = max_width / combined.shape[1]
                    combined = cv2.resize(
                        combined,
                        (int(combined.shape[1] * scale), int(combined.shape[0] * scale)),
                        interpolation=cv2.INTER_AREA,
                    )

                combined = self._draw_overlay(combined)
                cv2.imshow(self.window_name, combined)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    self.visualizer.reset_view()
        finally:
            self.close()

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        cv2.destroyAllWindows()
        self.visualizer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RT-MonoDepth pipeline from Jetson Nano Pi Camera feed",
    )
    parser.add_argument("--weight_path", required=True, help="Path to RT-MonoDepth weights directory")
    parser.add_argument("--config", default="config_jetson.json", help="Pipeline configuration file")
    parser.add_argument("--sensor_id", type=int, default=0, help="Camera sensor ID (0 for CSI lane 0)")
    parser.add_argument("--capture_width", type=int, default=1280, help="nvargus capture width")
    parser.add_argument("--capture_height", type=int, default=720, help="nvargus capture height")
    parser.add_argument("--display_width", type=int, default=640, help="Width after nvvidconv (appsink)")
    parser.add_argument("--display_height", type=int, default=360, help="Height after nvvidconv (appsink)")
    parser.add_argument("--fps", type=int, default=30, help="Camera framerate")
    parser.add_argument("--flip_method", type=int, default=0, help="nvvidconv flip-method (0..7)")
    parser.add_argument("--max_display_width", type=int, default=1920, help="Max width for preview window")
    parser.add_argument("--no_cuda", action="store_true", help="Force PyTorch inference on CPU")
    parser.add_argument("--no_downsample", action="store_true", help="Disable voxel downsampling in preprocessing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline = PiCameraPipeline(args)
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("Interrupted – shutting down.")
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"[ERROR] Pipeline crashed: {exc}", file=sys.stderr)
        return 1
    finally:
        pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

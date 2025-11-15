from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import PIL.Image as pil

from lib.gstreamer_utils import GStreamerCapture


def _escape_location(path: str) -> str:
    """Escape double quotes for GStreamer file location strings."""

    return path.replace('"', '\\"')


def _build_pipeline(source: Union[int, str], width: int, height: int) -> Tuple[str, str]:
    """Return a pipeline string and appsink name for the requested source."""

    appsink_name = "video_source_sink"
    if isinstance(source, int):
        pipeline = (
            f"v4l2src device=/dev/video{source} ! "
            "video/x-raw ! videoconvert ! videoscale ! "
            f"video/x-raw, width={width}, height={height}, format=BGR ! "
            "appsink name=video_source_sink max-buffers=1 drop=true sync=false"
        )
        return pipeline, appsink_name

    if isinstance(source, str):
        stripped = source.strip()
        if stripped.startswith("gst:"):
            return stripped[4:], appsink_name

        location = _escape_location(source)
        pipeline = (
            f"filesrc location=\"{location}\" ! decodebin ! videoconvert ! videoscale ! "
            f"video/x-raw, width={width}, height={height}, format=BGR ! "
            "appsink name=video_source_sink max-buffers=1 drop=true sync=false"
        )
        return pipeline, appsink_name

    raise TypeError(
        "VideoSource expects an int device index or string path/pipeline.")


class VideoSource:
    def __init__(self, source: Union[int, str], width: int = 640, height: int = 192):
        """Initialize a video source using GStreamer."""

        self.source = source
        self.width = width
        self.height = height
        self._capture: Optional[GStreamerCapture] = None

    def __enter__(self) -> "VideoSource":
        pipeline, sink_name = _build_pipeline(
            self.source, self.width, self.height)
        self._capture = GStreamerCapture(pipeline, appsink_name=sink_name)
        self._capture.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._capture is not None:
            self._capture.close()
            self._capture = None

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the video source."""

        if self._capture is None:
            return False, None

        success, frame = self._capture.read()
        if not success:
            return False, None

        return True, frame


def frame_to_tensor(frame: np.ndarray, target_size: Tuple[int, int]) -> pil.Image.Image:
    """Convert a BGR NumPy frame to an RGB PIL image of the target size."""

    rgb = np.ascontiguousarray(frame[..., ::-1])
    img = pil.fromarray(rgb)

    if img.size != target_size:
        img = img.resize(target_size, pil.LANCZOS)

    return img

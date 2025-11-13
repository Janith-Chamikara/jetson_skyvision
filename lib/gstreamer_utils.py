"""Lightweight GStreamer capture helpers for Jetson pipelines."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import gi
except ImportError as exc:  # pragma: no cover - environment check
    raise ImportError(
        "PyGObject (gi) is required for GStreamer capture. Install it with "
        "`sudo apt install python3-gi gir1.2-gst-plugins-base-1.0`."
    ) from exc

# Ensure required GStreamer components are present before importing
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp  # type: ignore

_GST_INITIALIZED = False


def ensure_gst_init() -> None:
    """Initialise GStreamer once per process."""

    global _GST_INITIALIZED
    if not _GST_INITIALIZED:
        Gst.init(None)
        _GST_INITIALIZED = True


class GStreamerCapture:
    """Minimal GStreamer appsink wrapper returning NumPy frames."""

    def __init__(self, pipeline: str, appsink_name: str = "appsink") -> None:
        self._pipeline_str = pipeline
        self._appsink_name = appsink_name
        self._pipeline: Optional[Gst.Pipeline] = None
        self._appsink: Optional[GstApp.AppSink] = None

    def open(self) -> None:
        """Construct and start the pipeline."""

        ensure_gst_init()
        self._pipeline = Gst.parse_launch(self._pipeline_str)
        element = self._pipeline.get_by_name(self._appsink_name)
        if element is None:
            raise RuntimeError(
                f"GStreamer pipeline missing appsink named '{self._appsink_name}'."
            )

        self._appsink = element  # type: ignore[assignment]
        self._appsink.set_property("emit-signals", False)
        self._appsink.set_property("sync", False)
        self._appsink.set_property("max-buffers", 1)
        self._appsink.set_property("drop", True)

        state_change = self._pipeline.set_state(Gst.State.PLAYING)
        if state_change == Gst.StateChangeReturn.FAILURE:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
            self._appsink = None
            raise RuntimeError("Failed to start GStreamer pipeline.")
        # Block until the state change completes so frames are ready to pull
        self._pipeline.get_state(Gst.CLOCK_TIME_NONE)

    def read(self, timeout_ns: int = Gst.SECOND) -> Tuple[bool, Optional[np.ndarray]]:
        """Retrieve the next frame from the appsink."""

        if self._appsink is None:
            return False, None

        sample = self._appsink.emit("try-pull-sample", timeout_ns)
        if sample is None:
            return False, None

        buffer = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")

        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            sample.unref()
            return False, None

        try:
            frame = np.frombuffer(map_info.data, dtype=np.uint8).reshape(
                (height, width, 3))
            return True, frame.copy()
        finally:
            buffer.unmap(map_info)
            sample.unref()

    def close(self) -> None:
        """Stop the pipeline and release resources."""

        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
            self._appsink = None

    def __enter__(self) -> "GStreamerCapture":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

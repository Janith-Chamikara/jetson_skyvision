# Jetson Nano Pi Camera Pipeline

This folder contains a lightweight entry-point for running the SkyVision depth + point-cloud pipeline directly on a Jetson Nano that uses the Raspberry Pi Camera V2 (CSI).

## Quick Start

1. Copy the minimal runtime bundle to the Jetson (see `prepare_jetson_bundle.py`).
2. Install dependencies on the Jetson (JetPack already provides CUDA, cuDNN, TensorRT):
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-opencv libopenblas-base
   pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v50 torch==2.1.0+nv23.10 torchvision==0.16.0+nv23.10
   pip3 install -r requirements_jetson.txt
   ```
3. Launch the pipeline:
   ```bash
   python3 jetson_nano/pi_camera_pointcloud.py --weight_path weights/RTMonoDepth --config config_jetson.json
   ```
4. Press `q` to exit, `r` to reset the Open3D viewpoint.

## Notes

- Update `config_jetson.json` with the calibrated intrinsics for your Pi camera (defaults are approximate).
- If you see a blank window, verify the camera ribbon cable and sensor ID (`--sensor_id`).
- On Nano 2GB models, consider lowering `--capture_width`/`--capture_height` to 960×540 for smoother performance.
- The avoidance overlay mirrors the desktop pipeline: the banner will turn red if the safety volume detects an imminent collision.

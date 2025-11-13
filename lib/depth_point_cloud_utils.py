import cv2
import numpy as np
import open3d as o3d
import torch

from layers import disp_to_depth
from networks.RTMonoDepth.RTMonoDepth import DepthDecoder, DepthEncoder


def compute_scaled_intrinsics(video_config, original_shape, depth_shape):
    """Scale camera intrinsics from calibration resolution to depth map resolution."""

    orig_h, orig_w = original_shape[:2]
    depth_h, depth_w = depth_shape

    calib_w = video_config.get(
        "calib_width", video_config.get("width", orig_w))
    calib_h = video_config.get(
        "calib_height", video_config.get("height", orig_h))

    scale_x = depth_w / float(calib_w)
    scale_y = depth_h / float(calib_h)

    fx = video_config["focal_length_x"] * scale_x
    fy = video_config["focal_length_y"] * scale_y
    cx = video_config["c_x"] * scale_x
    cy = video_config["c_y"] * scale_y

    return fx, fy, cx, cy


def build_pixel_grid(depth_shape):
    """Return pixel coordinate grid for back-projection."""

    height, width = depth_shape
    u_coords, v_coords = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32)
    )
    return u_coords, v_coords


def apply_transform(points, transform):
    """Apply a 4x4 homogeneous transform to a point cloud."""

    if transform is None:
        return points

    transform = np.asarray(transform, dtype=np.float32)
    if transform.shape != (4, 4):
        raise ValueError("Expected transform matrix with shape (4, 4)")

    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    hom_points = np.hstack((points, ones))
    transformed = hom_points @ transform.T
    return transformed[:, :3]


def depth_to_points(
    depth_map,
    color_rgb,
    intrinsics,
    min_depth,
    max_depth,
    pixel_grid=None,
    transform=None
):
    """Vectorised conversion from depth map and color image to XYZ points."""

    if depth_map.ndim != 2:
        raise ValueError("depth_map must be two dimensional (H x W)")

    fx, fy, cx, cy = intrinsics

    if pixel_grid is None:
        u_coords, v_coords = build_pixel_grid(depth_map.shape)
    else:
        u_coords, v_coords = pixel_grid

    valid_mask = np.isfinite(depth_map)
    if min_depth is not None:
        valid_mask &= depth_map >= min_depth
    if max_depth is not None:
        valid_mask &= depth_map <= max_depth

    if not np.any(valid_mask):
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32)
        )

    z = depth_map[valid_mask]
    u = u_coords[valid_mask]
    v = v_coords[valid_mask]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack((x, y, z), axis=-1).astype(np.float32)

    colors = color_rgb[valid_mask].reshape(-1, 3).astype(np.float32) / 255.0

    points = apply_transform(points, transform)

    return points, colors


def colorize_disparity(disp_map):
    """Convert a disparity map to a magma colormap following RT-MonoDepth scripts."""

    disp = np.asarray(disp_map, dtype=np.float32)
    if disp.ndim != 2:
        raise ValueError("disp_map must have shape H x W")

    valid_mask = np.isfinite(disp)
    if not np.any(valid_mask):
        return np.zeros((disp.shape[0], disp.shape[1], 3), dtype=np.uint8)

    finite_values = disp[valid_mask]
    vmin = float(finite_values.min())
    vmax = float(np.percentile(finite_values, 95))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(finite_values.max())
    if vmax <= vmin:
        vmax = vmin + 1e-6

    normalized = np.zeros_like(disp, dtype=np.float32)
    normalized[valid_mask] = (disp[valid_mask] - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0.0, 1.0)

    disp_uint8 = (normalized * 255.0).astype(np.uint8)
    colormap = cv2.applyColorMap(disp_uint8, cv2.COLORMAP_MAGMA)
    return colormap


class DepthProcessor:
    def __init__(
        self,
        weight_path,
        use_cuda=True,
        min_depth=0.1,
        max_depth=80.0,
        input_width=640,
        input_height=192
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_cuda else "cpu")
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.input_width = int(input_width)
        self.input_height = int(input_height)

        # Load models
        self.encoder = DepthEncoder()
        self.depth_decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc)

        # Load encoder
        encoder_dict = torch.load(
            f"{weight_path}/encoder.pth", map_location=self.device)
        self.encoder.load_state_dict(
            {k: v for k, v in encoder_dict.items() if k in self.encoder.state_dict()})
        self.encoder.to(self.device).eval()

        # Load decoder
        self.depth_decoder.load_state_dict(torch.load(
            f"{weight_path}/depth.pth", map_location=self.device))
        self.depth_decoder.to(self.device).eval()

    def estimate_depth(self, frame):
        """Run the network and return metric depth and disparity for a single frame."""

        with torch.no_grad():
            input_image = cv2.resize(
                frame, (self.input_width, self.input_height))
            input_tensor = (
                torch.from_numpy(input_image)
                .float()
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.device)
            ) / 255.0

            features = self.encoder(input_tensor)
            outputs = self.depth_decoder(features)
            disp = outputs[("disp", 0)]

            _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)
            depth_map = depth.squeeze().cpu().numpy().astype(np.float32)
            disp_map = disp.squeeze().cpu().numpy().astype(np.float32)

            return depth_map, input_image, disp_map


class PointCloudVisualizer:
    def __init__(self, config):
        """Initialize visualizer with configuration parameters."""

        self.point_cloud_config = config['point_cloud']
        self.vis_config = config['visualization']
        self.video_config = config['video']

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(
            window_name='RT-MonoDepth Point Cloud',
            width=self.point_cloud_config['window_width'],
            height=self.point_cloud_config['window_height']
        )
        self.point_cloud = o3d.geometry.PointCloud()

        opt = self.vis.get_render_option()
        opt.background_color = np.asarray(
            self.point_cloud_config['background_color'])
        opt.point_size = self.point_cloud_config['point_size']
        opt.show_coordinate_frame = self.vis_config['show_coordinate_frame']

        initial_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        initial_colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
        self.point_cloud.points = o3d.utility.Vector3dVector(initial_points)
        self.point_cloud.colors = o3d.utility.Vector3dVector(initial_colors)

        self.vis.add_geometry(self.point_cloud)
        self.reset_view()

        camera_to_body = self.point_cloud_config.get('camera_to_body')
        if camera_to_body is None:
            self._camera_to_body = np.eye(4, dtype=np.float32)
        else:
            try:
                camera_to_body = np.asarray(camera_to_body, dtype=np.float32)
                if camera_to_body.size != 16:
                    raise ValueError
                self._camera_to_body = camera_to_body.reshape(4, 4)
            except ValueError:
                print(
                    "Warning: Invalid camera_to_body transform; falling back to identity.")
                self._camera_to_body = np.eye(4, dtype=np.float32)

        self._intrinsics = None
        self._pixel_grid = None
        self._cached_shape = None

    def reset_view(self):
        view_control = self.vis.get_view_control()
        view_control.set_zoom(0.8)
        view_control.set_front([0, 0, -1])
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, -1, 0])

    def _ensure_geometry_cache(self, original_shape, depth_shape):
        if self._cached_shape == depth_shape:
            return

        self._intrinsics = compute_scaled_intrinsics(
            self.video_config, original_shape, depth_shape)
        self._pixel_grid = build_pixel_grid(depth_shape)
        self._cached_shape = depth_shape

    def create_point_cloud(self, rgb_image, depth_map):
        """Create point cloud from RGB image and depth map."""

        if depth_map.ndim != 2:
            raise ValueError("depth_map must have shape H x W")

        self._ensure_geometry_cache(rgb_image.shape[:2], depth_map.shape)

        rgb_for_colors = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        points, colors = depth_to_points(
            depth_map,
            rgb_for_colors,
            self._intrinsics,
            self.point_cloud_config.get('min_depth', 0.0),
            self.point_cloud_config.get('max_depth', None),
            pixel_grid=self._pixel_grid,
            transform=self._camera_to_body if self.point_cloud_config.get(
                'apply_extrinsics', False) else None
        )

        subsample = self.point_cloud_config.get('subsample_factor', 1)
        if subsample > 1 and len(points) > 0:
            points = points[::subsample]
            colors = colors[::subsample]

        return points, colors

    def update(self, points, colors):
        """Update point cloud visualization."""

        if len(points) > 0:
            self.point_cloud.points = o3d.utility.Vector3dVector(points)
            self.point_cloud.colors = o3d.utility.Vector3dVector(colors)
            self.vis.update_geometry(self.point_cloud)
            self.vis.poll_events()
            self.vis.update_renderer()

    def close(self):
        """Clean up resources."""

        self.vis.destroy_window()

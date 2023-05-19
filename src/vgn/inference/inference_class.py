import enum
import time
import torch
import dataclasses
import pathlib
import numpy as np
import rerun as rr
import spatialmath as sm
import open3d as o3d
from vgn.detection_implicit import VGNImplicit  # GIGA
from vgn.perception import TSDFVolume, create_tsdf
from vgn.ConvONets.conv_onet.generation import Generator3D


ZED2_INTRINSICS = np.array(
    [
        [1062.88232421875, 0.0, 957.660400390625],
        [0.0, 1062.88232421875, 569.8204345703125],
        [0.0, 0.0, 1.0],
    ]
)
ZED2_INTRINSICS_HALF = np.copy(ZED2_INTRINSICS)
ZED2_INTRINSICS_HALF[0:-1, :] /= 2
ZED2_INTRINSICS_HALF[1, 2] -= 14  # Cropping
ZED2_RESOLUTION = np.array([1920, 1080], dtype=np.int32)
ZED2_RESOLUTION_HALF = ZED2_RESOLUTION // 2
ZED2_RESOLUTION_HALF[1] -= 28  # Cropping

O_RESOLUTION = 40
O_SIZE = 0.3  # Meters --> I saw this somewhere in the repo
O_VOXEL_SIZE = O_SIZE / O_RESOLUTION


@dataclasses.dataclass
class State:
    tsdf: TSDFVolume


@dataclasses.dataclass
class CameraIntrinsic:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


INTRINSICS = CameraIntrinsic(
    width=ZED2_RESOLUTION_HALF[0],
    height=ZED2_RESOLUTION_HALF[1],
    fx=ZED2_INTRINSICS_HALF[0, 0],
    fy=ZED2_INTRINSICS_HALF[1, 1],
    cx=ZED2_INTRINSICS_HALF[0, 2],
    cy=ZED2_INTRINSICS_HALF[1, 2],
)


class ModelType(enum.Enum):
    packed = enum.auto()  # objects are placed on the table at their canonical poses.
    pile = enum.auto()  # objects are dropped on the workspace with random poses.


class GIGAInference:
    def __init__(
        self,
        model_dir: pathlib.Path,
        camera_intrinsic: CameraIntrinsic,
        model_type: ModelType = ModelType.packed,
        vis: bool = False,
    ):
        """
        ## Additional install stuff
        pip install .
        pip install cython
        pip install torch-scatter -f https://data.pyg.org/whl/{$TORCH}+${CUDA}.html # (check with conda list | grep torch)
        python scripts/convonet_setup.py build_ext --inplace
        pip install urdfpy
        pip install networkx==2.5
        """
        model_name = f"giga_{model_type.name}.pt"
        model_path = model_dir / model_name
        self.giga_model = VGNImplicit(
            model_path=model_path,
            model_type="giga",
            visualize=vis,
            # All arguments below are copied from the default list/arguments form the readme
            best=True,
            qual_th=0.75,
            force_detection=True,
            out_th=0.1,
            select_top=False,
        )
        self.camera_intrinsic = camera_intrinsic
        self.giga_model.net.eval()
        self.generator = Generator3D(
            self.giga_model.net,
            device=self.giga_model.device,
            input_type="pointcloud",
            padding=0,
        )
        self.O_SIZE = O_SIZE
        return

    def predict(self, depth: np.ndarray, camTtask_np: np.ndarray, reconstruction: bool = False):
        assert camTtask_np.shape == (4, 4)

        start_time = time.time()
        tsdf_volume = create_tsdf(
            size=O_SIZE,
            resolution=O_RESOLUTION,
            depth_imgs=np.expand_dims(depth, axis=0),
            intrinsic=self.camera_intrinsic,
            extrinsics=np.expand_dims(camTtask_np, axis=0),
        )

        state = State(tsdf=tsdf_volume)
        grasps, scores, toc = self.giga_model(state)
        inference_time = time.time() - start_time

        if reconstruction:
            pc_torch = torch.tensor(tsdf_volume.get_grid())
            pred_mesh, _ = self.generator.generate_mesh({"inputs": pc_torch})
        else:
            pred_mesh = None
        return grasps, scores, inference_time, tsdf_volume.get_cloud(), pred_mesh

    def visualize(self, grasp_mesh, wTcam, wTtask, tsdf_pc, rgb, depth, pred_mesh):
        rr.connect("192.168.0.120:9876")

        # Add images
        rr.log_image("rgb", rgb)
        rr.log_depth_image("depth", depth)

        # add_o3d_pointcloud
        o3d_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=self.camera_intrinsic.width,
            height=self.camera_intrinsic.height,
            fx=self.camera_intrinsic.fx,
            fy=self.camera_intrinsic.fy,
            cx=self.camera_intrinsic.cx,
            cy=self.camera_intrinsic.cy,
        )
        rgb_o3d = o3d.geometry.Image(rgb)
        depth_o3d = o3d.geometry.Image((depth * 1000).astype(np.uint16))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=False
        )
        full_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d_camera_intrinsic,
        )
        full_pcd_w = full_pcd.transform(wTcam.A)
        points = np.asanyarray(full_pcd_w.points)
        colors = np.asanyarray(full_pcd_w.colors) if full_pcd_w.has_colors() else None
        colors_uint8 = (colors * 255).astype(np.uint8) if full_pcd_w.has_colors() else None
        rr.log_points("giga/full_pcd_w", positions=points, colors=colors_uint8, radii=0.001)

        # Add origin
        rr.log_arrow(
            "giga/origin_x",
            origin=[0, 0, 0],
            vector=[0.1, 0, 0],
            color=[255, 0, 0],
            width_scale=0.01,
        )
        rr.log_arrow(
            "giga/origin_y",
            origin=[0, 0, 0],
            vector=[0, 0.1, 0],
            color=[0, 255, 0],
            width_scale=0.01,
        )
        rr.log_arrow(
            "giga/origin_z",
            origin=[0, 0, 0],
            vector=[0, 0, 0.1],
            color=[0, 0, 255],
            width_scale=0.01,
        )

        # add input_tsdf
        w_T_input_tsdf_pc = tsdf_pc.transform(wTtask.A)
        rr.log_points(
            "giga/input_tsdf_pc",
            positions=np.asanyarray(w_T_input_tsdf_pc.points),
            radii=0.001,
        )

        # add workspace obb
        ws_size = np.array([O_SIZE, O_SIZE, O_SIZE])
        rr.log_obb("giga/workspace", half_size=ws_size, position=wTtask.t + ws_size / 2)

        # Add grasps
        grasp_mesh_w = grasp_mesh.apply_transform(wTtask.A)
        rr.log_mesh(
            "giga/w_grasp",
            positions=grasp_mesh_w.vertices,
            indices=grasp_mesh_w.faces,
            normals=grasp_mesh_w.vertex_normals,
            vertex_colors=grasp_mesh_w.visual.vertex_colors,
        )

        if pred_mesh is None:
            return
        # Add shape
        scale_matrix = np.eye(4)
        scale_matrix[:3, :3] *= O_SIZE
        translation_matrix = sm.SE3.Trans(ws_size / 2).A
        pred_mesh.apply_transform(translation_matrix @ wTtask.A @ scale_matrix)
        rr.log_mesh(
            "giga/shape",
            positions=pred_mesh.vertices,
            indices=pred_mesh.faces,
            normals=pred_mesh.vertex_normals,
            vertex_colors=pred_mesh.visual.vertex_colors,
        )
        return

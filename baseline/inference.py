import enum
import time
import dataclasses
import pathlib
import trimesh
import torch
import numpy as np
import open3d as o3d
import spatialmath as sm
from vgn.detection_implicit import VGNImplicit  # GIGA
from vgn.perception import TSDFVolume, create_tsdf
from vgn.utils import visual
from vgn.ConvONets.conv_onet.generation import Generator3D


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


class ModelType(enum.Enum):
    packed = enum.auto()  # objects are placed on the table at their canonical poses.
    pile = enum.auto()  # objects are dropped on the workspace with random poses.


class GIGAInference:
    def __init__(
        self,
        model_dir: pathlib.Path,
        camera_intrinsic: o3d.camera.PinholeCameraIntrinsic,
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
            input_type='pointcloud',
            padding=0,
        )
        return

    def predict(self, rgb: np.ndarray, depth: np.ndarray, camera_pose: np.ndarray):
        assert camera_pose.shape == (4, 4)

        start_time = time.time()
        # TODO Make lists as input
        tsdf_volume = create_tsdf(
            size=O_SIZE,
            resolution=O_RESOLUTION,
            depth_imgs=np.expand_dims(depth, axis=0),
            intrinsic=self.camera_intrinsic,
            extrinsics=np.expand_dims(camera_pose, axis=0),
        )

        state = State(tsdf=tsdf_volume)
        grasps, scores, toc = self.giga_model(state)
        inference_time = time.time() - start_time
        return grasps, scores, inference_time, tsdf_volume
    
    def predict_mesh(self, pc_input):
        pred_mesh, _ = self.generator.generate_mesh({'inputs': pc_input})
        return pred_mesh


if __name__ == "__main__":
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

    intrinsics = CameraIntrinsic(
        width=ZED2_RESOLUTION_HALF[0],
        height=ZED2_RESOLUTION_HALF[1],
        fx=ZED2_INTRINSICS_HALF[0, 0],
        fy=ZED2_INTRINSICS_HALF[1, 1],
        cx=ZED2_INTRINSICS_HALF[0, 2],
        cy=ZED2_INTRINSICS_HALF[1, 2],
    )

    wTcam = sm.SE3(
        np.array(
            [
                [-0.00288795, -0.84104389, 0.54095919, 0.46337467],
                [-0.99974617, -0.00965905, -0.02035441, -0.07190939],
                [0.0223441, -0.54088066, -0.84080251, 1.06529043],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        check=False,
    )
    wTtask = sm.SE3.Trans([0.7, -0.24, 0.655])
    camTtask = wTcam.inv() * wTtask

    model_dir = pathlib.Path(__file__).parents[1] / "data/models"
    giga_inference = GIGAInference(model_dir, camera_intrinsic=intrinsics)

    # Load images
    # from PIL import Image

    # data_path = pathlib.Path(__file__).parents[1] / "baseline"
    # rgb_uint8_np = np.array(Image.open(data_path / "rgb.png"))
    # depth = np.array(Image.open(data_path / "depth.png"))
    # depth_np = depth.astype(np.float32) / 1000.0

    # Get images from camera
    from centergrasp_fmm.zed2 import ZED2Camera

    camera = ZED2Camera()
    rgb_uint8_np, depth_np, confidence_map_np = camera.get_image()

    # Do inference
    grasps, scores, inference_time, tsdf_volume = giga_inference.predict(
        rgb_uint8_np, depth_np, camTtask.A
    )
    pc_torch = torch.tensor(tsdf_volume.get_grid())
    pred_mesh = giga_inference.predict_mesh(pc_torch)

    # Visualize
    def pcd_from_rgbd(rgb, depth, project_valid_depth_only):
        o3d_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsics.width,
            height=intrinsics.height,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.cx,
            cy=intrinsics.cy,
        )
        rgb_o3d = o3d.geometry.Image(rgb)
        depth_o3d = o3d.geometry.Image((depth * 1000).astype(np.uint16))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=False
        )
        full_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d_camera_intrinsic,
            project_valid_depth_only=project_valid_depth_only,
        )
        return full_pcd

    full_pcd = pcd_from_rgbd(rgb_uint8_np, depth_np, project_valid_depth_only=True)
    full_pcd_w = full_pcd.transform(wTcam)
    grasp_mesh_list = [visual.grasp2mesh(g, s) for g, s in zip(grasps, scores)]
    grasp_mesh_list = [grasp_mesh_list[0]] if len(grasp_mesh_list) > 0 else None

    import rerun as rr

    # rr.init("centergrasp")
    # rr.spawn()
    rr.connect("192.168.0.120:9876")

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
    w_T_input_tsdf_pc = tsdf_volume.get_cloud().transform(wTtask.A)
    rr.log_points(
        "giga/input_tsdf_pc",
        positions=np.asanyarray(w_T_input_tsdf_pc.points),
        radii=0.001,
    )

    # add workspace obb
    ws_size = np.array([O_SIZE, O_SIZE, O_SIZE])
    rr.log_obb("giga/workspace", half_size = ws_size , position=wTtask.t + ws_size / 2)

    # add_o3d_pointcloud
    points = np.asanyarray(full_pcd_w.points)
    colors = np.asanyarray(full_pcd_w.colors) if full_pcd_w.has_colors() else None
    colors_uint8 = (colors * 255).astype(np.uint8) if full_pcd_w.has_colors() else None
    rr.log_points("giga/full_pcd_w", positions=points, colors=colors_uint8, radii=0.001)

    # Add grasps
    w_grasp_mesh_list = [
        grasp.apply_transform(wTtask.A) for grasp in grasp_mesh_list
    ]
    for i, mesh in enumerate(w_grasp_mesh_list):
        rr.log_mesh(
            "giga/w_grasp" + f"_{i}",
            positions=mesh.vertices,
            indices=mesh.faces,
            normals=mesh.vertex_normals,
            vertex_colors=mesh.visual.vertex_colors,
        )

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
        vertex_colors=pred_mesh.visual.vertex_colors
    )

    # Add images
    rr.log_image("rgb", rgb_uint8_np)
    rr.log_depth_image("depth", depth_np)

    print("Done")

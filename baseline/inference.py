import enum
import time
import dataclasses
import pathlib
import trimesh
import numpy as np
import open3d as o3d
from vgn.detection_implicit import VGNImplicit  # GIGA
from vgn.perception import TSDFVolume, create_tsdf

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
        model_type: ModelType = ModelType.pile,
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
            qual_th=0.9,
            force_detection=True,
            out_th=0.1,
            select_top=False,
        )
        self.giga_model.net.eval()
        self.camera_intrinsic = camera_intrinsic

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

        # Make empty object
        state = State(tsdf=tsdf_volume)
        # Create point cloud in grasping frame
        # scene_mesh = trimesh.PointCloud(np.asarray(state.tsdf.get_cloud()))
        # world_scene_mesh = trimesh.PointCloud(np.asarray(state.tsdf._volume.get_point_cloud()[:, :3]))
        grasps, scores, toc, composed_scene = self.giga_model(state)
        inference_time = time.time() - start_time
        return grasps, scores, inference_time, composed_scene

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

    wTcam = np.array(
            [
                [-0.00288795, -0.84104389, 0.54095919, 0.46337467],
                [-0.99974617, -0.00965905, -0.02035441, -0.07190939],
                [0.0223441, -0.54088066, -0.84080251, 1.06529043],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    wTcam[:3, 3] += np.array([-0.7, 0.2, -0.6])
    camTw = np.linalg.inv(wTcam)


    model_dir = pathlib.Path(__file__).parents[1] / "data/models"
    giga_inference = GIGAInference(
            model_dir, camera_intrinsic=intrinsics, vis=True
        )

    from PIL import Image

    data_path = pathlib.Path(__file__).parents[1] / "baseline"
    rgb_uint8 = np.array(Image.open(data_path / "rgb.png"))
    depth = np.array(Image.open(data_path / "depth.png"))
    depth = depth.astype(np.float32) / 1000.0

    grasps, scores, inference_time, composed_scene = giga_inference.predict(rgb_uint8, depth, camTw)


    # Visualize
    import rerun as rr
    rr.init("centergrasp")
    rr.spawn()
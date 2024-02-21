import pathlib
import numpy as np
from PIL import Image
from typing import Tuple
from vgn.grasp import Grasp
from vgn.utils.transform import Transform
from vgn.perception import CameraIntrinsic
from graspnetAPI import GraspNet
from graspnetAPI.utils.xmlhandler import xmlReader
from graspnetAPI.utils.utils import parse_posevector, generate_views
from graspnetAPI.utils.rotation import batch_viewpoint_params_to_matrix


KINECT_K = np.array(
    [[631.54864502, 0.0, 638.43517329], [0.0, 631.20751953, 366.49904066], [0.0, 0.0, 1.0]]
)
KINECT_INTRINSIC = CameraIntrinsic(1280, 720, 631.54864502, 631.20751953, 638.43517329, 366.49904066)
GRASPNET_ROOT = pathlib.Path.home() / "datasets/graspnet"
GIGA_GRASPNET_ROOT = pathlib.Path.home() / "datasets/centergrasp_g/giga_graspnet"


def get_obj_poses(pose_fpath: str) -> Tuple[np.ndarray]:
    scene_reader = xmlReader(pose_fpath)
    posevector = scene_reader.getposevectorlist()
    obj_idx, poses = zip(*[parse_posevector(vec) for vec in posevector])
    obj_idx = np.array(obj_idx)
    poses = np.array(poses)
    # Sort by obj_idx, to align with the saved binary masks
    ind_argsort = np.argsort(obj_idx)
    obj_idx = obj_idx[ind_argsort]
    poses = poses[ind_argsort]
    return obj_idx, poses


class GraspNetReader:
    def __init__(self, mode: str = "train") -> None:
        self.camera = "kinect"
        self.graspnet_api = GraspNet(root=GRASPNET_ROOT, camera=self.camera, split=mode)
        return

    def __len__(self):
        return len(self.graspnet_api)

    def get_scene_img_names(self, idx: int) -> Tuple[str]:
        rgbPath = pathlib.Path(self.graspnet_api.loadData(idx)[0])
        scene_name = rgbPath.parents[2].name
        img_name = rgbPath.stem
        return scene_name, img_name

    def get_data_np(self, idx: int):
        paths = self.graspnet_api.loadData(idx)
        rgbPath = pathlib.Path(paths[0])
        depthPath = pathlib.Path(paths[1])
        camera_poses_path = rgbPath.parents[1] / "camera_poses.npy"
        cam0_wrt_table_path = rgbPath.parents[1] / "cam0_wrt_table.npy"
        poses_path = str(rgbPath).replace("/rgb/", "/annotations/").replace(".png", ".xml")
        img_idx = int(rgbPath.stem)

        rgb = np.array(Image.open(rgbPath)) # uint8
        depth = np.array(Image.open(depthPath), dtype=np.float32) / 1000
        obj_indices, poses = get_obj_poses(poses_path)
        rel_cam_poses = np.load(camera_poses_path)
        cam0_wrt_table = np.load(cam0_wrt_table_path)
        cam_pose = cam0_wrt_table @ rel_cam_poses[img_idx]
        return rgb, depth, poses, obj_indices, cam_pose

    def load_scene_grasps(self, idx: int) -> np.ndarray:
        scene_name, img_name = self.get_scene_img_names(idx)
        sceneId, annId = int(scene_name.replace("scene_", "")), int(img_name)
        grasp_group = self.graspnet_api.loadGrasp(sceneId, annId, fric_coef_thresh=1.0)
        # TODO: did we forget widths in our sgdf?
        return
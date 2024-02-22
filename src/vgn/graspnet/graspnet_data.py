import pathlib
import numpy as np
from PIL import Image
from typing import Tuple
from vgn.grasp import Grasp
from vgn.utils.transform import Transform
from vgn.perception import CameraIntrinsic
from graspnetAPI import GraspNet, GraspGroup
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
    
    @property
    def scenes_idx(self):
        return self.graspnet_api.sceneIds

    def __len__(self):
        return len(self.graspnet_api)
    
    def load_depth(self, scene_idx: int, img_idx: int) -> np.ndarray:
        depth = self.graspnet_api.loadDepth(sceneId=scene_idx, camera=self.camera, annId=img_idx)
        return depth
    
    def load_cam_pose(self, scene_idx: int, img_idx: int) -> np.ndarray:
        paths = self.graspnet_api.loadData(scene_idx, self.camera, img_idx)
        rgbPath = pathlib.Path(paths[0])
        cam0_wrt_table_path = rgbPath.parents[1] / "cam0_wrt_table.npy"
        cam0_wrt_table = np.load(cam0_wrt_table_path)
        if img_idx == 0:
            return cam0_wrt_table
        camera_poses_path = rgbPath.parents[1] / "camera_poses.npy"
        rel_cam_poses = np.load(camera_poses_path)
        cam_pose = cam0_wrt_table @ rel_cam_poses[img_idx]
        return cam_pose
    
    def load_obj_poses(self, scene_idx: int) -> Tuple[np.ndarray]:
        img_idx = 0
        paths = self.graspnet_api.loadData(scene_idx, self.camera, img_idx)
        obj_poses_path = paths[0].replace("/rgb/", "/annotations/").replace(".png", ".xml")
        obj_indices, camTposes = get_obj_poses(obj_poses_path)
        wTcam = self.load_cam_pose(scene_idx, img_idx)
        obj_poses = [wTcam @ camTpose for camTpose in camTposes]
        return obj_indices, obj_poses

    def load_scene_grasps(self, scene_idx: int) -> GraspGroup:
        wTcam = self.load_cam_pose(scene_idx, img_idx=0)
        grasp_group = self.graspnet_api.loadGrasp(scene_idx, annId=0, fric_coef_thresh=1.0)
        grasp_group = grasp_group.transform(wTcam)
        return grasp_group

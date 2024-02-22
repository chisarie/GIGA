import numpy as np
import open3d as o3d
from tqdm import tqdm

from graspnetAPI import Grasp as GraspNetGrasp
from vgn.grasp import Grasp as VGNGrasp
from vgn.grasp import Label as VGNLabel
from vgn.io import *
from vgn.perception import *
from vgn.utils.transform import Rotation, Transform

from vgn.graspnet.graspnet_data import KINECT_INTRINSIC, GraspNetReader, GIGA_GRASPNET_ROOT, GRASPNET_ROOT


def grasp_graspnet_to_vgn(grasp: GraspNetGrasp) -> VGNGrasp:
    orientation = Rotation.from_matrix(grasp.rotation_matrix)
    position = grasp.translation
    width = grasp.width
    return VGNGrasp(Transform(orientation, position), width)

def main(visualize: bool = False):
    np.random.seed(123)
    finger_depth = 0.05
    max_opening_width = 0.08
    size = 6 * finger_depth
    camera_intrinsic = KINECT_INTRINSIC

    data_reader = GraspNetReader()

    (GIGA_GRASPNET_ROOT / "scenes").mkdir(parents=True, exist_ok=True)
    (GIGA_GRASPNET_ROOT / "mesh_pose_list").mkdir(parents=True, exist_ok=True)
    write_setup(
        GIGA_GRASPNET_ROOT,
        size,
        camera_intrinsic,
        max_opening_width,
        finger_depth,
    )

    for scene_idx in tqdm(range(len(data_reader.scenes_idx))):
        # Depth images (i.e. scenes)
        depth_img = data_reader.load_depth(scene_idx, img_idx=0)
        cam_pose = data_reader.load_cam_pose(scene_idx, img_idx=0)
        extrinsic = np.r_[Rotation.from_matrix(cam_pose[:3, :3]).as_quat(), cam_pose[:3, 3]]
        # TODO: check depth_img format (uint vs float)
        scene_uuid = write_sensor_data(GIGA_GRASPNET_ROOT, depth_img, extrinsic)

        # Mesh pose list
        obj_indices, obj_poses = data_reader.load_obj_poses(scene_idx)
        mesh_pose_list = []
        for obj_idx, obj_pose in zip(obj_indices, obj_poses):
            scale = 1.0
            mesh_path = GRASPNET_ROOT / "models" / f"{obj_idx:03d}" / "textured.obj"
            mesh_pose_list.append((str(mesh_path), scale, obj_pose))
        write_point_cloud(GIGA_GRASPNET_ROOT, scene_uuid, mesh_pose_list, name="mesh_pose_list")
        
        # Grasps
        scene_grasp_group = data_reader.load_scene_grasps(scene_idx)
        for grasp in scene_grasp_group:
            vgn_grasp = grasp_graspnet_to_vgn(grasp)
            label = VGNLabel.SUCCESS if grasp.score >= 0.5 else VGNLabel.FAILURE
            write_grasp(GIGA_GRASPNET_ROOT, scene_uuid, vgn_grasp, label)
        if visualize:
            # depth_o3d = o3d.geometry.Image((depth_img * 1000).astype(np.uint16))
            depth_o3d = o3d.geometry.Image(depth_img)
            o3d_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=camera_intrinsic.width,
                height=camera_intrinsic.height,
                fx=camera_intrinsic.fx,
                fy=camera_intrinsic.fy,
                cx=camera_intrinsic.cx,
                cy=camera_intrinsic.cy,
            )
            full_pcd = o3d.geometry.PointCloud.create_from_depth_image(
                depth_o3d,
                o3d_camera_intrinsic,
                np.linalg.inv(cam_pose),
            )
            origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            cam_pose_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1).transform(cam_pose)
            obj_poses_o3d = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1).transform(obj_pose) for obj_pose in obj_poses]
            grasp_indices = np.random.choice(len(scene_grasp_group), size=50, replace=False)
            o3d_grasp_meshes = [grasp.to_open3d_geometry() for grasp in scene_grasp_group[grasp_indices]]
            o3d.visualization.draw_geometries([full_pcd, origin, cam_pose_o3d, *obj_poses_o3d, *o3d_grasp_meshes])
    return


if __name__ == "__main__":
    main(visualize=True)

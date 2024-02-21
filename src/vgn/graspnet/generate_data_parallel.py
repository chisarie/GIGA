import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import scipy.signal as signal
from tqdm import tqdm
import multiprocessing as mp

from vgn.grasp import Grasp, Label
from vgn.io import *
from vgn.perception import *
from vgn.utils.transform import Rotation, Transform
from vgn.utils.implicit import get_mesh_pose_list_from_world

from vgn.graspnet.graspnet_data import KINECT_INTRINSIC, GraspNetReader, GIGA_GRASPNET_ROOT, GRASPNET_ROOT


def main(args, rank):
    np.random.seed()
    seed = np.random.randint(0, 1000) + rank
    np.random.seed(seed)

    data_reader = GraspNetReader()
    finger_depth = 0.05
    max_opening_width = 0.08
    size = 6 * finger_depth
    ws_lower = np.array([0.02, 0.02, 0.0])
    ws_upper = np.array([size-0.02, size-0.02, size])
    camera_intrinsic = KINECT_INTRINSIC


    (GIGA_GRASPNET_ROOT / "scenes").mkdir(parents=True, exist_ok=True)
    (GIGA_GRASPNET_ROOT / "mesh_pose_list").mkdir(parents=True, exist_ok=True)
    write_setup(
        GIGA_GRASPNET_ROOT,
        size,
        camera_intrinsic,
        max_opening_width,
        finger_depth,
    )

    for idx in tqdm(range(len(data_reader))):
        _, depth, poses, obj_indices, cam_pose = data_reader.get_data_np(idx)

        # reconstrct point cloud using a subset of the images
        # tsdf = create_tsdf(size, 120, depth_imgs, sim.camera.intrinsic, extrinsics)
        # pc = tsdf.get_cloud()

        # crop surface and borders from point cloud
        # bounding_box = o3d.geometry.AxisAlignedBoundingBox(ws_lower, ws_upper)
        # pc = pc.crop(bounding_box)
        # o3d.visualization.draw_geometries([pc])

        # if pc.is_empty():
        #     print("Point cloud empty, skipping scene")
        #     continue

        # store the raw data
        scene_id = write_sensor_data(GIGA_GRASPNET_ROOT, depth, cam_pose)

        mesh_pose_list = []
        for i, obj_idx in enumerate(obj_indices):
            mesh_path = GRASPNET_ROOT / "models" / f"{obj_idx:03d}" / "textured.obj"
            scale = 1.0
            obj_pose = poses[i]
            mesh_pose_list.append((str(mesh_path), scale, obj_pose))
        write_point_cloud(GIGA_GRASPNET_ROOT, scene_id, mesh_pose_list, name="mesh_pose_list")

        grasps_success, grasps_failure = data_reader.load_scene_grasps(idx)
        for grasp in grasps_success:
            label = Label.SUCCESS
            write_grasp(GIGA_GRASPNET_ROOT, scene_id, grasp, label)
        for grasp in grasps_failure:
            label = Label.FAILURE
            write_grasp(GIGA_GRASPNET_ROOT, scene_id, grasp, label)
    return


def evaluate_grasp_point(sim, pos, normal, num_rotations=6):
    # define initial grasp frame on object surface
    z_axis = -normal
    x_axis = np.r_[1.0, 0.0, 0.0]
    if np.isclose(np.abs(np.dot(x_axis, z_axis)), 1.0, 1e-4):
        x_axis = np.r_[0.0, 1.0, 0.0]
    y_axis = np.cross(z_axis, x_axis)
    x_axis = np.cross(y_axis, z_axis)
    R = Rotation.from_matrix(np.vstack((x_axis, y_axis, z_axis)).T)

    # try to grasp with different yaw angles
    yaws = np.linspace(0.0, np.pi, num_rotations)
    outcomes, widths = [], []
    for yaw in yaws:
        ori = R * Rotation.from_euler("z", yaw)
        sim.restore_state()
        candidate = Grasp(Transform(ori, pos), width=sim.gripper.max_opening_width)
        outcome, width = sim.execute_grasp(candidate, remove=False)
        outcomes.append(outcome)
        widths.append(width)

    # detect mid-point of widest peak of successful yaw angles
    # TODO currently this does not properly handle periodicity
    successes = (np.asarray(outcomes) == Label.SUCCESS).astype(float)
    if np.sum(successes):
        peaks, properties = signal.find_peaks(
            x=np.r_[0, successes, 0], height=1, width=1
        )
        idx_of_widest_peak = peaks[np.argmax(properties["widths"])] - 1
        ori = R * Rotation.from_euler("z", yaws[idx_of_widest_peak])
        width = widths[idx_of_widest_peak]

    return Grasp(Transform(ori, pos), width), int(np.max(outcomes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--save-scene", action="store_true")
    args = parser.parse_args()
    args.save_scene = True
    if args.num_proc > 1:
        pool = mp.Pool(processes=args.num_proc)
        for i in range(args.num_proc):
            pool.apply_async(func=main, args=(args, i))
        pool.close()
        pool.join()
    else:
        main(args, 0)

import pathlib
import numpy as np
import spatialmath as sm
from PIL import Image
from vgn.utils import visual
from vgn.inference.inference_class import GIGAInference, INTRINSICS

def main():
    # Load data
    data_path = pathlib.Path(__file__).parent
    wTcam_np = np.load(data_path / "wTcam_np.npy")
    wTtask_np = np.load(data_path / "wTtask_np.npy")
    rgb_uint8 = np.array(Image.open(data_path / "rgb.png"))
    depth = np.array(Image.open(data_path / "depth.png"))
    depth = depth.astype(np.float32) / 1000.0
    model_dir = pathlib.Path(__file__).parents[3] / "data/models"

    # Initialize
    wTcam = sm.SE3(wTcam_np, check=False)
    wTtask = sm.SE3(wTtask_np, check=False)
    camTtask = wTcam.inv() * wTtask
    giga_inference = GIGAInference(model_dir, camera_intrinsic=INTRINSICS)
    
    # Do inference
    grasps, scores, inference_time, tsdf_pc, pred_mesh = giga_inference.predict(
        depth, camTtask.A, reconstruction=False
    )
    best_grasp = sm.SE3.Rt(R=grasps[0].pose.rotation.as_matrix(), t=grasps[0].pose.translation)
    wTgrasp = wTtask * best_grasp

    # Visualize
    grasp_mesh_list = [visual.grasp2mesh(g, s) for g, s in zip(grasps, scores)]
    grasp_mesh = grasp_mesh_list[0] if len(grasp_mesh_list) > 0 else None
    giga_inference.visualize(grasp_mesh, wTcam, wTtask, tsdf_pc, rgb_uint8, depth, pred_mesh)

    # Save output
    np.save(data_path / "inference_time.npy", inference_time)
    np.save(data_path / "wTgrasp_np.npy", wTgrasp.A)
    return


if __name__ == "__main__":
    main()

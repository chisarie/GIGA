import pathlib
import numpy as np
from PIL import Image
import spatialmath as sm
from vgn.inference.inference_class import GIGAInference, INTRINSICS


# Get Image dirs
real_dir = pathlib.Path(__file__).parents[6] / "datasets/rgbd_table/real/train_pbr/000000"
depth_paths = sorted(real_dir.glob("depth/*.png"))
giga_mesh_dir = real_dir / "giga_mesh"
giga_mesh_dir.mkdir(exist_ok=True)

# Get GIGA model
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
model_dir = pathlib.Path(__file__).parents[3] / "data/models"

# Initialize
camTtask = wTcam.inv() * wTtask
giga_inference = GIGAInference(model_dir, camera_intrinsic=INTRINSICS)

for depth_path in depth_paths:
    depth_uint16 = np.array(Image.open(depth_path))
    depth_np = depth_uint16.astype(np.float32) / 1000.0
    # Do inference
    grasps, scores, inference_time, tsdf_pc, pred_mesh = giga_inference.predict(
        depth_np, camTtask.A, reconstruction=True
    )
    # Save mesh
    pred_mesh_path = giga_mesh_dir / (depth_path.stem + ".ply")
    pred_mesh.export(pred_mesh_path)
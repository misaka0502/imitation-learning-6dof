"""Instantiate FurnitureSim-v0 and test various functionalities."""
try:
    import isaacgym
    from isaacgym import gymapi, gymtorch
except ImportError as e:
    from rich import print

    print(
        """[red][Isaac Gym Import Error]
        1. You need to install Isaac Gym, if not installed.
            - Download Isaac Gym following https://clvrai.github.io/furniture-bench/docs/getting_started/installation_guide_furniture_sim.html#download-isaac-gym
            - Then, pip install -e isaacgym/python
        2. If PyTorch was imported before furniture_bench, please import torch after furniture_bench.[/red]
        """
            )
    print()
    raise ImportError(e)
import argparse
import pickle

import furniture_bench

import gym
import cv2
import torch
import numpy as np
from src.gym import get_env
from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv
from furniture_bench.envs.observation import DEFAULT_VISUAL_OBS, DEFAULT_STATE_OBS, FULL_OBS
import sys
sys.path.append("/home2/zxp/Projects/Juicer_ws/imitation-juicer/foundationpose")
from foundationpose.estimater import *
from foundationpose.datareader import *
from foundationpose.Utils import *
from foundationpose.learning.training.predict_score import *
from foundationpose.learning.training.predict_pose_refine import *
from datetime import datetime
from src.common.context import suppress_all_output
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import time
import torch
import numpy as np
import collections
from furniture_bench.config import config
from furniture_bench.utils.pose import get_mat
import furniture_bench.controllers.control_utils as C
import time

ROBOT_HEIGHT = 0.015
table_pos = np.array([0.8, 0.8, 0.4])
table_half_width = 0.015
table_surface_z = table_pos[2] + table_half_width
franka_pose = np.array(
    [0.5 * -table_pos[0] + 0.1, 0, table_surface_z + ROBOT_HEIGHT]
)
base_tag_from_robot_mat = config["robot"]["tag_base_from_robot_base"]
franka_from_origin_mat = get_mat(
    [franka_pose[0], franka_pose[1], franka_pose[2]],
    [0, 0, 0],
)

def sim_to_april_mat():
    return torch.tensor(
        np.linalg.inv(base_tag_from_robot_mat) @ np.linalg.inv(franka_from_origin_mat),
        device="cpu", dtype=torch.float64
    )

def sim_coord_to_april_coord(sim_coord_mat):
    return sim_to_april_mat() @ sim_coord_mat

def cam_coord_to_april_coord(pose_est_cam, cam_pos, cam_target):
    cam_pos = np.array(cam_pos)
    cam_target = np.array(cam_target)
    z_camera = (cam_target - cam_pos) / np.linalg.norm(cam_target - cam_pos)
    up_axis = np.array([0, 0, 1])  # Assuming Z is the up axis
    x_camera = np.cross(up_axis, z_camera)
    x_camera /= np.linalg.norm(x_camera)
    y_camera = np.cross(z_camera, x_camera)
    R_camera_sim = np.vstack([x_camera, y_camera, z_camera]).T
    T_camera_sim = np.eye(4)
    T_camera_sim[:3, :3] = R_camera_sim
    T_camera_sim[:3, 3] = cam_pos
    pose_est_cam = pose_est_cam * np.array([-1, -1, 1, 1]).reshape(4, -1)
    pos_est_sim = T_camera_sim @ pose_est_cam
    pose_est_april_coord = np.concatenate(
        [
            *C.mat2pose(
                sim_coord_to_april_coord(
                    torch.tensor(pos_est_sim, device="cpu", dtype=torch.float64)
                )
            )
        ]
    )
    return pose_est_april_coord

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--furniture", default="square_table")
    parser.add_argument(
        "--file-path", help="Demo path to replay (data directory or pickle)"
    )
    parser.add_argument(
        "--scripted", action="store_true", help="Execute hard-coded assembly script."
    )
    parser.add_argument("--no-action", action="store_true")
    parser.add_argument("--random-action", action="store_true")
    parser.add_argument(
        "--input-device",
        help="Device to control the robot.",
        choices=["keyboard", "oculus", "keyboard-oculus"],
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--init-assembled",
        action="store_true",
        help="Initialize the environment with the assembled furniture.",
    )
    parser.add_argument(
        "--save-camera-input",
        action="store_true",
        help="Save camera input of the simulator at the beginning of the episode.",
    )
    parser.add_argument(
        "--record", action="store_true", help="Record the video of the simulator."
    )
    parser.add_argument(
        "--high-res",
        action="store_true",
        help="Use high resolution images for the camera input.",
    )
    parser.add_argument(
        "--randomness",
        default="low",
        help="Randomness level of the environment.",
    )
    parser.add_argument(
        "--high-random-idx",
        default=0,
        type=int,
        help="The index of high_randomness.",
    )
    parser.add_argument(
        "--env-id",
        default="FurnitureSim-v0",
        help="Environment id of FurnitureSim",
    )
    parser.add_argument(
        "--replay-path", type=str, help="Path to the saved data to replay action."
    )

    parser.add_argument(
        "--act-rot-repr",
        type=str,
        help="Rotation representation for action space.",
        choices=["quat", "axis", "rot_6d"],
        default="quat",
    )

    parser.add_argument(
        "--compute-device-id",
        type=int,
        default=0,
        help="GPU device ID used for simulation.",
    )

    parser.add_argument(
        "--graphics-device-id",
        type=int,
        default=0,
        help="GPU device ID used for rendering.",
    )

    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--pose", action="store_true")
    args = parser.parse_args()

    # Create FurnitureSim environment.
    env = gym.make(
        args.env_id,
        furniture=args.furniture,
        num_envs=args.num_envs,
        resize_img=not args.high_res,
        obs_keys=DEFAULT_VISUAL_OBS + ["parts_poses"] + ["depth_image2"] + ["color_image4"] + ["depth_image4"],
        init_assembled=args.init_assembled,
        record=args.record,
        headless=args.headless,
        save_camera_input=args.save_camera_input,
        randomness=args.randomness,
        high_random_idx=args.high_random_idx,
        act_rot_repr=args.act_rot_repr,
        ctrl_mode='osc',
        compute_device_id=args.compute_device_id,
        graphics_device_id=args.graphics_device_id,
    )
    # env: FurnitureSimEnv = get_env(
    #     gpu_id=args.gpu,
    #     furniture=args.furniture,
    #     num_envs=args.n_envs,
    #     randomness=args.randomness,
    #     max_env_steps=5_000,
    #     resize_img=False,
    #     act_rot_repr="rot_6d",
    #     ctrl_mode="osc",
    #     action_type=args.action_type,
    #     april_tags=not args.no_april_tags,
    #     verbose=args.verbose,
    #     headless=not args.visualize,
    #     record=args.record
    # )

    # Initialize FurnitureSim.
    ob = env.reset()
    done = False
    if args.pose:
        print("Pose estimation is enabled!!!")
        time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        code_dir = "/home2/zxp/Projects/Juicer_ws/imitation-juicer"
        # mesh_file = f'{code_dir}/foundationpose/demo_data/square_table/mesh/square_table.obj'
        mesh_file = f'{code_dir}/foundationpose/demo_data/square_table_leg/mesh/square_table_leg4.obj'
        # test_scene_dir = f'{code_dir}/foundationpose/demo_data/square_table'
        test_scene_dir = f'{code_dir}/foundationpose/demo_data/square_table_leg'
        debug_dir = f'{code_dir}/foundationpose/debug/run_sim_env/{time_now}'
        debug = 2
        set_logging_format()
        set_seed(0)
        mesh = trimesh.load(mesh_file, force='mesh')
        # mesh2 = trimesh.load(mesh_file2, force='mesh')
        if debug>=2:
            os.system(f'mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')
        if debug >= 1:
            os.system(f'mkdir -p {debug_dir}/rollouts_vis {debug_dir}/rollouts_ob')
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        # to_origin2, extents2 = trimesh.bounds.oriented_bounds(mesh2)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
        # bbox2 = np.stack([-extents2/2, extents2/2], axis=0).reshape(2,3)
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
        # est2 = FoundationPose(model_pts=mesh2.vertices, model_normals=mesh2.vertex_normals, mesh=mesh2, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
        logging.info("estimator initialization done")
        reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=None, zfar=np.inf)
        # reader2 = YcbineoatReader(video_dir=test_scene_dir2, shorter_side=None, zfar=np.inf)
        step_idx = 0
        with suppress_all_output(True):
            color_img = ob["color_image2"][0]
            color = reader.get_color(color_img)
            depth_img = ob["depth_image2"][0] * 1000
            depth_img = depth_img.astype(np.uint16)
            depth_img = 65535 - depth_img
            depth = reader.get_depth(depth_img)
            mask = reader.get_mask(0).astype(bool)
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=5)
            # mask2 = reader2.get_mask(1).astype(bool)
            # pose2 = est2.register(K=reader2.K, rgb=color, depth=depth, ob_mask=mask2, iteration=5)
            # if debug>=3:
            #     m = mesh.copy()
            #     m.apply_transform(pose)
            #     m.export(f'{debug_dir}/model_tf.obj')
            #     xyz_map = depth2xyzmap(depth, reader.K)
            #     valid = depth>=0.001
            #     pcd = toOpen3dCloud(xyz_map[valid], color[valid])
            #     o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
            
            center_pose = pose@np.linalg.inv(to_origin)
            # center_pose2 = pose2@np.linalg.inv(to_origin2)
            if debug>=1:
                # os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
                # os.makedirs(f'{debug_dir}/ob_in_cam_apriltag_sim', exist_ok=True)
                vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
                # vis = draw_posed_3d_box(reader2.K, img=color, ob_in_cam=center_pose2, bbox=bbox2)
                # vis = draw_xyz_axis(vis, ob_in_cam=center_pose2, scale=0.1, K=reader2.K, thickness=3, transparency=0, is_input_rgb=True)
                # np.savetxt(f"{debug_dir}/begin_parts_poses_leg.txt", ob["parts_poses"][:,-7:].cpu().numpy())
                # pil_color_image2 = to_pil_image(ob["color_image2"].squeeze(0).permute(2, 0, 1))
                # pil_color_image2.save(f"{debug_dir}/rollouts_vis/begin_raw.png")
                # cv2.imwrite(f'{debug_dir}/rollouts_vis/begin_raw.png', color_img)
                # np.savetxt(f'{debug_dir}/rollouts_ob/begin_pose.txt', pose.reshape(4,4))
                # cv2.imshow('1', vis[...,::-1])
                # cv2.waitKey(1)
            if debug>=2:
                os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
                cv2.imwrite(f'{debug_dir}/track_vis/{step_idx:019d}.png', vis)
            # six_dof_poses = [torch.from_numpy(pose).cpu()]
            # parts_poses = [ob["parts_poses"].cpu()]

    def action_tensor(ac):
        if isinstance(ac, (list, np.ndarray)):
            return torch.tensor(ac).float().to(env.device)

        ac = ac.clone()
        if len(ac.shape) == 1:
            ac = ac[None]
        return ac.tile(args.num_envs, 1).float().to(env.device)

    # Rollout one episode with a selected policy:
    if args.input_device is not None:
        # Teleoperation.
        device_interface = furniture_bench.device.make(args.input_device)

        while not done:
            action, _ = device_interface.get_action()
            action = action_tensor(action)
            ob, rew, done, _ = env.step(action)

    elif args.no_action or args.init_assembled:
        # Execute 0 actions.
        while True:
            if args.act_rot_repr == "quat":
                ac = action_tensor([0, 0, 0, 0, 0, 0, 1, -1])
            else:
                ac = action_tensor([0, 0, 0, 0, 0, 0, -1])
            ob, rew, done, _ = env.step(ac)
    elif args.random_action:
        # Execute randomly sampled actions.
        import tqdm

        pbar = tqdm.tqdm()
        while True:
            ac = action_tensor(env.action_space.sample())
            ob, rew, done, _ = env.step(ac)
            pbar.update(args.num_envs)

    elif args.file_path is not None:
        # Play actions in the demo.
        with open(args.file_path, "rb") as f:
            data = pickle.load(f)
        for ac in data["actions"]:
            ac = action_tensor(ac)
            env.step(ac)
    elif args.scripted:
        # Execute hard-coded assembly script.
        while not done:
            action, skill_complete = env.get_assembly_action()
            action = action_tensor(action)
            ob, rew, done, _ = env.step(action)
            step_idx += 1
            if args.pose:
                step_idx += 1
                with suppress_all_output(True):
                    color_img = ob["color_image2"][0]
                    color = reader.get_color(color_img)
                    depth_img = ob["depth_image2"][0] * 1000
                    depth_img = depth_img.astype(np.uint16)
                    depth_img = 65535 - depth_img
                    depth = reader.get_depth(depth_img)
                    pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=2)
                    # pose2 = est2.track_one(rgb=color, depth=depth, K=reader2.K, iteration=2)
                    if debug>=1:
                        center_pose = pose@np.linalg.inv(to_origin)
                        # center_pose2 = pose2@np.linalg.inv(to_origin2)
                        vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
                        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
                        # vis = draw_posed_3d_box(reader2.K, img=color, ob_in_cam=center_pose2, bbox=bbox2)
                        # vis = draw_xyz_axis(vis, ob_in_cam=center_pose2, scale=0.1, K=reader2.K, thickness=3, transparency=0, is_input_rgb=True)
                        # np.savetxt(f'{debug_dir}/ob_in_cam/{step_idx}.txt', pose.reshape(4,4))
                        # np.savetxt(f'{debug_dir}/ob_in_cam_apriltag_sim/{step_idx}.txt', ob["parts_poses"][:, -7:].cpu().numpy())
                        # cv2.imshow('1', vis[...,::-1])
                        # cv2.waitKey(1)
                    if debug>=2:
                        os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
                        imageio.imwrite(f'{debug_dir}/track_vis/{step_idx:019d}.png', vis)
                    # six_dof_poses.append(torch.from_numpy(pose).cpu())
                    # parts_poses.append(ob["parts_poses"].cpu())

        if args.pose:
            cv2.imwrite(f'{debug_dir}/rollouts_vis/end.png', vis)
            # np.savetxt(f'{debug_dir}/rollouts_ob/end_pose.txt', pose.reshape(4,4))
            # np.savetxt(f"{debug_dir}/parts_poses_leg.txt", parts_poses[:, -7:].numpy())
            print("done!!!")
            # print(ob.keys())
            # print(ob["color_image2"].squeeze(0).cpu().numpy().shape)
            # cv2.imshow("1", ob["color_image2"].squeeze(0).cpu().numpy())
            # cv2.waitKey(1)
    elif args.replay_path:
        # Replay the trajectory.
        with open(args.replay_path, "rb") as f:
            data = pickle.load(f)
        env.reset_to([data["observations"][0]])  # reset to the first observation.
        for ac in data["actions"]:
            ac = action_tensor(ac)
            ob, rew, done, _ = env.step(ac)
    else:
        raise ValueError(f"No action specified")

    print("done")


if __name__ == "__main__":
    main()

# """Instantiate FurnitureSim-v0 and test various functionalities."""

# import argparse
# import pickle

# import furniture_bench

# import gym
# import cv2
# import torch
# import numpy as np


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--furniture", default="square_table")
#     parser.add_argument(
#         "--file-path", help="Demo path to replay (data directory or pickle)"
#     )
#     parser.add_argument(
#         "--scripted", action="store_true", help="Execute hard-coded assembly script."
#     )
#     parser.add_argument("--no-action", action="store_true")
#     parser.add_argument("--random-action", action="store_true")
#     parser.add_argument(
#         "--input-device",
#         help="Device to control the robot.",
#         choices=["keyboard", "oculus", "keyboard-oculus"],
#     )
#     parser.add_argument("--headless", action="store_true")
#     parser.add_argument(
#         "--init-assembled",
#         action="store_true",
#         help="Initialize the environment with the assembled furniture.",
#     )
#     parser.add_argument(
#         "--save-camera-input",
#         action="store_true",
#         help="Save camera input of the simulator at the beginning of the episode.",
#     )
#     parser.add_argument(
#         "--record", action="store_true", help="Record the video of the simulator."
#     )
#     parser.add_argument(
#         "--high-res",
#         action="store_true",
#         help="Use high resolution images for the camera input.",
#     )
#     parser.add_argument(
#         "--randomness",
#         default="low",
#         help="Randomness level of the environment.",
#     )
#     parser.add_argument(
#         "--high-random-idx",
#         default=0,
#         type=int,
#         help="The index of high_randomness.",
#     )
#     parser.add_argument(
#         "--env-id",
#         default="FurnitureSim-v0",
#         help="Environment id of FurnitureSim",
#     )
#     parser.add_argument(
#         "--replay-path", type=str, help="Path to the saved data to replay action."
#     )

#     parser.add_argument(
#         "--act-rot-repr",
#         type=str,
#         help="Rotation representation for action space.",
#         choices=["quat", "axis", "rot_6d"],
#         default="quat",
#     )

#     parser.add_argument(
#         "--compute-device-id",
#         type=int,
#         default=0,
#         help="GPU device ID used for simulation.",
#     )

#     parser.add_argument(
#         "--graphics-device-id",
#         type=int,
#         default=0,
#         help="GPU device ID used for rendering.",
#     )

#     parser.add_argument("--num-envs", type=int, default=1)
#     args = parser.parse_args()

#     # Create FurnitureSim environment.
#     env = gym.make(
#         args.env_id,
#         furniture=args.furniture,
#         num_envs=args.num_envs,
#         resize_img=not args.high_res,
#         init_assembled=args.init_assembled,
#         record=args.record,
#         headless=args.headless,
#         save_camera_input=args.save_camera_input,
#         randomness=args.randomness,
#         high_random_idx=args.high_random_idx,
#         act_rot_repr=args.act_rot_repr,
#         compute_device_id=args.compute_device_id,
#         graphics_device_id=args.graphics_device_id,
#     )
#     # Initialize FurnitureSim.
#     ob = env.reset()
#     done = False

#     def action_tensor(ac):
#         if isinstance(ac, (list, np.ndarray)):
#             return torch.tensor(ac).float().to(env.device)

#         ac = ac.clone()
#         if len(ac.shape) == 1:
#             ac = ac[None]
#         return ac.tile(args.num_envs, 1).float().to(env.device)

#     # Rollout one episode with a selected policy:
#     if args.input_device is not None:
#         # Teleoperation.
#         device_interface = furniture_bench.device.make(args.input_device)

#         while not done:
#             action, _ = device_interface.get_action()
#             action = action_tensor(action)
#             ob, rew, done, _ = env.step(action)

#     elif args.no_action or args.init_assembled:
#         # Execute 0 actions.
#         while True:
#             if args.act_rot_repr == "quat":
#                 ac = action_tensor([0, 0, 0, 0, 0, 0, 1, -1])
#             else:
#                 ac = action_tensor([0, 0, 0, 0, 0, 0, -1])
#             ob, rew, done, _ = env.step(ac)
#     elif args.random_action:
#         # Execute randomly sampled actions.
#         import tqdm

#         pbar = tqdm.tqdm()
#         while True:
#             ac = action_tensor(env.action_space.sample())
#             ob, rew, done, _ = env.step(ac)
#             pbar.update(args.num_envs)

#     elif args.file_path is not None:
#         # Play actions in the demo.
#         with open(args.file_path, "rb") as f:
#             data = pickle.load(f)
#         for ac in data["actions"]:
#             ac = action_tensor(ac)
#             env.step(ac)
#     elif args.scripted:
#         # Execute hard-coded assembly script.
#         while not done:
#             action, skill_complete = env.get_assembly_action()
#             action = action_tensor(action)
#             ob, rew, done, _ = env.step(action)
#     elif args.replay_path:
#         # Replay the trajectory.
#         with open(args.replay_path, "rb") as f:
#             data = pickle.load(f)
#         env.reset_to([data["observations"][0]])  # reset to the first observation.
#         for ac in data["actions"]:
#             ac = action_tensor(ac)
#             ob, rew, done, _ = env.step(ac)
#     else:
#         raise ValueError(f"No action specified")

#     print("done")


# if __name__ == "__main__":
#     main()

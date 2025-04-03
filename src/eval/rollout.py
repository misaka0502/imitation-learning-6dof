import furniture_bench
from omegaconf import DictConfig  # noqa: F401
import torch

import collections

import numpy as np
from tqdm import tqdm, trange
from ipdb import set_trace as bp  # noqa: F401
from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv

from typing import Union

from src.behavior.base import Actor
from src.visualization.render_mp4 import create_in_memory_mp4
from src.common.context import suppress_all_output
from src.common.tasks import furniture2idx
from src.common.files import trajectory_save_dir
from src.data_collection.io import save_raw_rollout
from src.data_processing.utils import resize, resize_crop

import wandb
import time
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import os
import sys
sys.path.append("/home2/zxp/Projects/Juicer_ws/imitation-juicer/foundationpose")
from foundationpose.estimater import *
from foundationpose.datareader import *
from foundationpose.Utils import *
from foundationpose.learning.training.predict_score import *
from foundationpose.learning.training.predict_pose_refine import *
from datetime import datetime

from furniture_bench.config import config
from furniture_bench.utils.pose import get_mat
import furniture_bench.controllers.control_utils as C

RolloutStats = collections.namedtuple(
    "RolloutStats",
    [
        "success_rate",
        "n_success",
        "n_rollouts",
        "epoch_idx",
        "rollout_max_steps",
        "total_return",
        "total_reward",
    ],
)

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
save_flag = True
def sim_to_april_mat():
    return torch.tensor(
        np.linalg.inv(base_tag_from_robot_mat) @ np.linalg.inv(franka_from_origin_mat),
        device="cpu", dtype=torch.float64
    )

def sim_coord_to_april_coord(sim_coord_mat):
    return sim_to_april_mat() @ sim_coord_mat

def cam_coord_to_april_coord(pose_est_cam, cam_pos, cam_target, revise=False):
    cam_pos = np.array(cam_pos)
    cam_target = np.array(cam_target)
    z_camera = (cam_target - cam_pos) / np.linalg.norm(cam_target - cam_pos)
    up_axis = np.array([0, 0, 1])  # Assuming Z is the up axis
    x_camera = -np.cross(up_axis, z_camera)
    x_camera /= np.linalg.norm(x_camera)
    y_camera = np.cross(z_camera, x_camera)
    R_camera_sim = np.vstack([x_camera, y_camera, z_camera]).T
    T_camera_sim = np.eye(4)
    T_camera_sim[:3, :3] = R_camera_sim
    T_camera_sim[:3, 3] = cam_pos
    pos_est_sim = T_camera_sim @ pose_est_cam
    if revise:
        r_y = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        pos_est_sim[:3, :3] = pos_est_sim[:3, :3] @ r_y
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

def april_to_sim_mat():
        return franka_from_origin_mat @ base_tag_from_robot_mat
def april_coord_to_sim_coord(april_coord_mat):
        """Converts AprilTag coordinate to simulator base_tag coordinate."""
        return april_to_sim_mat() @ april_coord_mat

def april_coord_to_cam_coord(pose_est_april, cam_pos, cam_target):
    cam_pos = np.array(cam_pos)
    cam_target = np.array(cam_target)
    z_camera = (cam_target - cam_pos) / np.linalg.norm(cam_target - cam_pos)
    up_axis = np.array([0, 0, 1])  # Assuming Z is the up axis
    x_camera = -np.cross(up_axis, z_camera)
    x_camera /= np.linalg.norm(x_camera)
    y_camera = np.cross(z_camera, x_camera)
    R_camera_sim = np.vstack([x_camera, y_camera, z_camera]).T
    T_camera_sim = np.eye(4)
    T_camera_sim[:3, :3] = R_camera_sim
    T_camera_sim[:3, 3] = cam_pos
    pose_est_april = torch.tensor(pose_est_april, device="cpu", dtype=torch.float64)
    pose_est_april_coord = april_coord_to_sim_coord(
                    C.pose2mat(pose_est_april[:3], pose_est_april[-4:],  device="cpu").numpy()
                )
    pose_est_april_coord = np.linalg.inv(T_camera_sim) @ pose_est_april_coord
    return pose_est_april_coord

def rollout(
    env: FurnitureSimEnv,
    actor: Actor,
    rollout_max_steps: int,
    iter: int,
    pbar: tqdm = None,
    resize_video: bool = True,
    ests:FoundationPose = None,
    readers:YcbineoatReader = None,
    to_origins = None,
    bboxs = None,
    debug = 0,
    debug_dir = None,
    num_poses=1
):
    # os.system(f'rm -rf {debug_dir}/rollouts_vis/* {debug_dir}/rollouts_ob/* && mkdir -p {debug_dir}/rollouts_vis {debug_dir}/rollouts_ob')
    os.makedirs(f'{debug_dir}/rollouts_vis/leg/rgb', exist_ok=True)
    # os.makedirs(f'{debug_dir}/rollouts_vis/leg/depth', exist_ok=True)
    # os.makedirs(f'{debug_dir}/rollouts_vis/top/rgb', exist_ok=True)
    # os.makedirs(f'{debug_dir}/rollouts_vis/top/depth', exist_ok=True)
    # os.makedirs(f'{debug_dir}/rollouts_ob/leg', exist_ok=True)
    # os.makedirs(f'{debug_dir}/rollouts_ob/top', exist_ok=True)
    global save_flag
    camera_idx = ["2", "4"]
    # get first observation
    with suppress_all_output(False):
        obs = env.reset()

    actor.action_refresh()

    # print(f'obs shape: {obs["robot_state"].shape}')
    if env.furniture_name == "lamp":
        # Before we start, let the environment settle by doing nothing for 1 second
        for _ in range(50):
            obs, reward, done, _ = env.step_noop()

    video_obs = obs.copy()

    step_idx = 0

    if readers is not None:
        with suppress_all_output(True):
            colors = [readers[i].get_color(obs[f"color_image{camera_idx[i]}"][0]) for i in range(num_poses)]
            depths = [readers[i].get_depth(65535-(obs[f"depth_image{camera_idx[i]}"][0] * 1000).astype(np.uint16)) for i in range(num_poses)]
            masks = [readers[i].get_mask(0).astype(bool) for i in range(num_poses)]
            poses_begin = [ests[i].register(K=readers[i].K, rgb=colors[i], depth=depths[i], ob_mask=masks[i], iteration=5) for i in range(num_poses)]

            if debug >= 1:
                center_poses = [poses_begin[i]@np.linalg.inv(to_origins[i]) for i in range(num_poses)]
                vis_begins = [draw_posed_3d_box(readers[i].K, img=colors[i], ob_in_cam=center_poses[i], bbox=bboxs[i]) for i in range(num_poses)]
                # center_pose = pose_begin@np.linalg.inv(to_origin)
                # center_pose2 = pose_begin2@np.linalg.inv(to_origin2)
                # vis_begin = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
                # vis_begin = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
                cv2.imwrite(f'{debug_dir}/rollouts_vis/leg/rgb/{iter:019d}_begin.png', vis_begins[0])
                # vis_begin2 = draw_posed_3d_box(reader2.K, img=color2, ob_in_cam=center_pose2, bbox=bbox2)
                # vis_begin2 = draw_xyz_axis(color2, ob_in_cam=center_pose2, scale=0.1, K=reader2.K, thickness=3, transparency=0, is_input_rgb=True)
                cv2.imwrite(f'{debug_dir}/rollouts_vis/top/rgb/{iter:019d}_begin.png', vis_begins[1])
    
    # cv2.imwrite(f'{debug_dir}/rollouts_vis/leg/{iter:019d}_begin.png', obs["color_image2"].squeeze(0).cpu().numpy())
    # Resize the images in the observation
    obs["color_image1"][0] = resize(obs["color_image1"][0])
    obs["color_image2"][0] = resize_crop(obs["color_image2"][0])

    obs_horizon = actor.obs_horizon
    if readers is not None:
        pose_begin_april_coord = cam_coord_to_april_coord(poses_begin[0], [0.90, -0.00, 0.65], [-1, -0.00, 0.3])
        obs["parts_poses"][:, -7:] = torch.tensor(pose_begin_april_coord, device=env.device)
        pose_begin_april_coord = cam_coord_to_april_coord(poses_begin[1], [0.3, -0.65, 0.8], [0.3, 0.8, 0.00], revise=True)
        print(f"6dpe: {pose_begin_april_coord}")
        print(f"apriltag: {obs['parts_poses'][:, :7]}")
        obs["parts_poses"][:, :7] = torch.tensor(pose_begin_april_coord, device=env.device)

    obs_deque = collections.deque(
        [obs] * obs_horizon,
        maxlen=obs_horizon,
    )
    if resize_video:
        video_obs["color_image1"][0] = resize(video_obs["color_image1"][0])
        video_obs["color_image2"][0] = resize_crop(video_obs["color_image2"][0])

    # save visualization and rewards
    robot_states = [video_obs["robot_state"].cpu()]
    imgs1 = [video_obs["color_image1"][0].cpu()]
    imgs2 = [video_obs["color_image2"][0].cpu()]
    depths = [video_obs["depth_image2"][0].cpu()]
    parts_poses = [video_obs["parts_poses"].cpu()]
    actions = list()
    rewards = list()
    done = torch.zeros((env.num_envs, 1), dtype=torch.bool, device="cuda")

    while not done.all():
        # Get the next actions from the actor
        action_pred = actor.action(obs_deque)

        obs, reward, done, _ = env.step(action_pred)

        video_obs = obs.copy()

        if readers is not None:
            with suppress_all_output(True):
                colors = [readers[i].get_color(obs[f"color_image{camera_idx[i]}"][0]) for i in range(num_poses)]
                depths = [readers[i].get_depth(65535-(obs[f"depth_image{camera_idx[i]}"][0] * 1000).astype(np.uint16)) for i in range(num_poses)]
                masks = [readers[i].get_mask(0).astype(bool) for i in range(num_poses)]
                poses = [ests[i].track_one(rgb=colors[i], depth=depths[i], K=readers[i].K, iteration=2) for i in range(num_poses)]

                if save_flag:
                    center_poses = [poses[i]@np.linalg.inv(to_origins[i]) for i in range(num_poses)]
                    vis = [draw_posed_3d_box(readers[i].K, img=colors[i], ob_in_cam=center_poses[i], bbox=bboxs[i]) for i in range(num_poses)]
                    imageio.imwrite(f'{debug_dir}/track_vis/leg/{step_idx:019d}.png', vis[0])
                    imageio.imwrite(f'{debug_dir}/track_vis/top/{step_idx:019d}.png', vis[1])
                if debug>=2:
                    center_poses = [poses[i]@np.linalg.inv(to_origins[i]) for i in range(num_poses)]
                    vis = [draw_posed_3d_box(readers[i].K, img=colors[i], ob_in_cam=center_poses[i], bbox=bboxs[i]) for i in range(num_poses)]
                    # center_pose = pose@np.linalg.inv(to_origin)
                    # center_pose2 = pose2@np.linalg.inv(to_origin2)
                    # vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
                    # vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
                    # vis2 = draw_posed_3d_box(reader2.K, img=color2, ob_in_cam=center_pose2, bbox=bbox2)
                    # vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
                    # vis = draw_xyz_axis(color2, ob_in_cam=center_pose2, scale=0.1, K=reader2.K, thickness=3, transparency=0, is_input_rgb=True)
                    # cv2.imshow('1', vis[...,::-1])
                    # cv2.waitKey(1)
                if debug>=3:
                    os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
                    imageio.imwrite(f'{debug_dir}/track_vis/leg/{step_idx:019d}.png', vis[0])
                    os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
                    imageio.imwrite(f'{debug_dir}/track_vis/top/{step_idx:019d}.png', vis[1])
        if save_flag:
            cv2.imwrite(f'{debug_dir}/rollouts_vis/leg/rgb/{step_idx:019d}.png', obs["color_image2"][0])
        image_end = obs["color_image2"][0]
        if resize_video:
            video_obs["color_image1"][0] = resize(video_obs["color_image1"][0])
            video_obs["color_image2"][0] = resize_crop(video_obs["color_image2"][0])
        obs["color_image1"][0] = resize(obs["color_image1"][0])
        obs["color_image2"][0] = resize_crop(obs["color_image2"][0])
        # Save observations for the policy
        if readers is not None:
            pose_april_coord = cam_coord_to_april_coord(poses[0], [0.90, -0.00, 0.65], [-1, -0.00, 0.3])
            obs["parts_poses"][:, -7:] = torch.tensor(pose_april_coord, device=env.device)
            pose_april_coord = cam_coord_to_april_coord(poses[1], [0.3, -0.65, 0.8], [0.3, 0.8, 0.00], revise=True)
            obs["parts_poses"][:, :7] = torch.tensor(pose_april_coord, device=env.device)
        obs_deque.append(obs)

        # Store the results for visualization and logging
        robot_states.append(video_obs["robot_state"].cpu())
        imgs1.append(video_obs["color_image1"][0])
        imgs2.append(video_obs["color_image2"][0])
        depths.append(video_obs["depth_image2"][0])
        actions.append(action_pred.cpu())
        rewards.append(reward.cpu())
        parts_poses.append(video_obs["parts_poses"].cpu())

        # update progress bar
        step_idx += 1
        if pbar is not None:
            pbar.set_postfix(step=step_idx)
            pbar.update()

        if step_idx >= rollout_max_steps:
            done = torch.ones((env.num_envs, 1), dtype=torch.bool, device="cuda")

        if done.all():
            save_flag = False
            cv2.imwrite(f'{debug_dir}/rollouts_vis/leg/{iter:019d}_end.png', image_end)
            if readers is not None:
                center_poses = [poses[i]@np.linalg.inv(to_origins[i]) for i in range(num_poses)]
                vis_ends = [draw_posed_3d_box(readers[i].K, img=colors[i], ob_in_cam=center_poses[i], bbox=bboxs[i]) for i in range(num_poses)]
                # center_pose = pose@np.linalg.inv(to_origin)
                # center_pose2 = pose2@np.linalg.inv(to_origin2)
                # pose_end_april_coord = cam_coord_to_april_coord(pose, [0.90, -0.00, 0.65], [-1, -0.00, 0.3])
                # vis_end = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
                # vis_end = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
                # vis_end2 = draw_posed_3d_box(reader.K, img=color2, ob_in_cam=center_pose2, bbox=bbox2)
                # vis_end2 = draw_xyz_axis(color2, ob_in_cam=center_pose2, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
                cv2.imwrite(f'{debug_dir}/rollouts_vis/leg/rgb/{iter:019d}_end.png', vis_ends[0])
                cv2.imwrite(f'{debug_dir}/rollouts_vis/top/rgb/{iter:019d}_end.png', vis_ends[1])
            #     # np.savetxt(f'{debug_dir}/rollouts_ob/{iter:019d}_end.txt', pose_end_april_coord)
            break
    if readers is not None:
        return (
            torch.stack(robot_states, dim=1),
            torch.stack(imgs1, dim=1),
            torch.stack(imgs2, dim=1),
            torch.stack(actions, dim=1),
            # Using cat here removes the singleton dimension
            torch.cat(rewards, dim=1),
            torch.stack(parts_poses, dim=1),
        )
    else:
        return (
            torch.stack(robot_states, dim=1),
            torch.stack(imgs1, dim=1),
            torch.stack(imgs2, dim=1),
            torch.stack(actions, dim=1),
            # Using cat here removes the singleton dimension
            torch.cat(rewards, dim=1),
            torch.stack(parts_poses, dim=1)
        )


@torch.no_grad()
def calculate_success_rate(
    env: FurnitureSimEnv,
    actor: Actor,
    n_rollouts: int,
    rollout_max_steps: int,
    epoch_idx: int,
    gamma: float = 0.99,
    rollout_save_dir: Union[str, None] = None,
    save_failures: bool = False,
    n_parts_assemble: Union[int, None] = None,
    compress_pickles: bool = False,
    resize_video: bool = True,
    pose_estimation: bool = False, # Whether to use pose estimation
    num_poses: int = 1, # Number of objects to estimate
) -> RolloutStats:
    def pbar_desc(self: tqdm, i: int, n_success: int):
        rnd = i + 1
        total = rnd * env.num_envs
        success_rate = n_success / total if total > 0 else 0
        self.set_description(
            f"Performing rollouts ({env.furniture_name}): round {rnd}/{n_rollouts//env.num_envs}, success: {n_success}/{total} ({success_rate:.1%})"
        )
    
    scorer:ScorePredictor = None
    refiner:PoseRefinePredictor = None
    glctx:dr.RasterizeCudaContext = None
    est:FoundationPose = None
    reader:YcbineoatReader = None
    debug_dir:str = None
    mesh = None
    to_origin = None
    bbox = None
    debug = 0

    # Set FoundationPose arguments
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    code_dir = "/home2/zxp/Projects/Juicer_ws/imitation-juicer"
    debug_dir = f'{code_dir}/foundationpose/debug/{time_now}'
    os.system(f'rm -rf {debug_dir}/rollouts_vis/* {debug_dir}/rollouts_ob/* && mkdir -p {debug_dir}/rollouts_vis {debug_dir}/rollouts_ob')
    if pose_estimation:
        print("Pose estimation is enabled!!!")
        debug = 1
        set_logging_format()
        set_seed(0)
        os.system(f'mkdir -p {debug_dir}/track_vis/leg {debug_dir}/track_vis/top')
        if debug>=2:
            os.system(f'mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')
        if debug >= 1:
            os.system(f'mkdir -p {debug_dir}/rollouts_vis {debug_dir}/rollouts_ob')
        
        mesh_files = []
        test_scene_dirs = []
        mesh_files.append(f'{code_dir}/foundationpose/demo_data/square_table_leg/mesh/square_table_leg4.obj')
        mesh_files.append(f'{code_dir}/foundationpose/demo_data/square_table/mesh/square_table.obj')
        test_scene_dirs.append(f'{code_dir}/foundationpose/demo_data/square_table_leg')
        test_scene_dirs.append(f'{code_dir}/foundationpose/demo_data/square_table')
        # mesh_file = f'{code_dir}/foundationpose/demo_data/square_table_leg/mesh/square_table_leg4.obj'
        # test_scene_dir = f'{code_dir}/foundationpose/demo_data/square_table_leg'
        # mesh_file2 = f'{code_dir}/foundationpose/demo_data/square_table/mesh/square_table.obj'
        # test_scene_dir2 = f'{code_dir}/foundationpose/demo_data/square_table'
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        meshs = []
        to_origins = []
        extents = []
        bboxs = []
        ests = []
        readers = []
        for i in range(num_poses):
            meshs.append(trimesh.load(mesh_files[i], force='mesh'))
            to_origin, extent = trimesh.bounds.oriented_bounds(meshs[i])
            to_origins.append(to_origin)
            extents.append(extent)
            bboxs.append(np.stack([-extent/2, extent/2], axis=0).reshape(2,3))
            ests.append(FoundationPose(model_pts=meshs[i].vertices, model_normals=meshs[i].vertex_normals, mesh=meshs[i], scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx))
            readers.append(YcbineoatReader(video_dir=test_scene_dirs[i], shorter_side=None, zfar=np.inf))
        
        # mesh = trimesh.load(mesh_file, force='mesh')
        # mesh2 = trimesh.load(mesh_file2, force='mesh')
        # to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        # to_origin2, extents2 = trimesh.bounds.oriented_bounds(mesh2)
        # bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
        # bbox2 = np.stack([-extents2/2, extents2/2], axis=0).reshape(2,3)
        # scorer = ScorePredictor()
        # refiner = PoseRefinePredictor()
        # glctx = dr.RasterizeCudaContext()
        # est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
        # est2 = FoundationPose(model_pts=mesh2.vertices, model_normals=mesh2.vertex_normals, mesh=mesh2, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
        # logging.info("estimator initialization done")
        # reader = YcbineoatReader(video_dir=test_scene_dir, shorter_side=None, zfar=np.inf)
        # reader2 = YcbineoatReader(video_dir=test_scene_dir2, shorter_side=None, zfar=np.inf)

    if n_parts_assemble is None:
        n_parts_assemble = len(env.furniture.should_be_assembled)

    tbl = wandb.Table(
        columns=["rollout", "success", "epoch", "reward", "return", "steps"]
    )
    pbar = trange(
        n_rollouts,
        desc="Performing rollouts",
        leave=False,
        total=rollout_max_steps * (n_rollouts // env.num_envs),
    )

    tqdm.pbar_desc = pbar_desc

    n_success = 0

    all_robot_states = list()
    all_imgs1 = list()
    all_imgs2 = list()
    all_actions = list()
    all_rewards = list()
    all_parts_poses = list()
    all_success = list()

    pbar.pbar_desc(0, n_success)
    for i in range(n_rollouts // env.num_envs):
        # Perform a rollout with the current model
        if pose_estimation:
            robot_states, imgs1, imgs2, actions, rewards, parts_poses = rollout(
                env,
                actor,
                rollout_max_steps,
                iter=i,
                pbar=pbar,
                resize_video=resize_video,
                ests=ests,
                readers=readers,
                to_origins=to_origins,
                bboxs=bboxs,
                debug=debug,
                debug_dir=debug_dir,
                num_poses=num_poses
            )
        else:
            robot_states, imgs1, imgs2, actions, rewards, parts_poses = rollout(
                env,
                actor,
                rollout_max_steps,
                iter=i,
                pbar=pbar,
                resize_video=resize_video,
                debug=debug,
                debug_dir=debug_dir,
            )

        # Calculate the success rate
        success = rewards.sum(dim=1) == n_parts_assemble
        # if debug == 1:
        #     cv2.imwrite(f'{debug_dir}/rollouts_vis/{i:019d}_begin.png', vis_begin)
        #     np.savetxt(f'{debug_dir}/rollouts_ob/{i:019d}_begin.txt', pose_begin)
        #     cv2.imwrite(f'{debug_dir}/rollouts_vis/{i:019d}_end.png', vis_end)
        #     np.savetxt(f'{debug_dir}/rollouts_ob/{i:019d}_end.txt', pose_end)
        #     np.savetxt(f"{debug_dir}/leg_poses.txt", parts_poses.squeeze(0)[:, -7:].numpy())
        #     np.savetxt(f"{debug_dir}/parts_poses.txt", parts_poses.squeeze(0).numpy())

        n_success += success.sum().item()

        # Save the results from the rollout
        all_robot_states.extend(robot_states)
        all_imgs1.extend(imgs1)
        all_imgs2.extend(imgs2)
        all_actions.extend(actions)
        all_rewards.extend(rewards)
        all_parts_poses.extend(parts_poses)
        all_success.extend(success)

        # Update the progress bar
        pbar.pbar_desc(i, n_success)

    total_return = 0
    total_reward = 0
    table_rows = []
    for rollout_idx in trange(n_rollouts, desc="Saving rollouts", leave=False):
        # Get the rewards and images for this rollout
        robot_states = all_robot_states[rollout_idx].numpy()
        video1 = all_imgs1[rollout_idx].numpy()
        video2 = all_imgs2[rollout_idx].numpy()
        actions = all_actions[rollout_idx].numpy()
        rewards = all_rewards[rollout_idx].numpy()
        parts_poses = all_parts_poses[rollout_idx].numpy()
        success = all_success[rollout_idx].item()
        furniture = env.furniture_name

        # Number of steps until success, i.e., the index of the final reward received
        n_steps = np.where(rewards == 1)[0][-1] + 1 if success else rollout_max_steps

        # Stack the two videos side by side into a single video
        # and keep axes as (T, H, W, C) (and cut off after rollout reaches success)
        video = np.concatenate([video1, video2], axis=2)[:n_steps]
        video = create_in_memory_mp4(video, fps=20)

        # Calculate the reward and return for this rollout
        total_reward += np.sum(rewards)
        episode_return = np.sum(rewards * gamma ** np.arange(len(rewards)))
        total_return += episode_return

        table_rows.append(
            [
                wandb.Video(video, fps=20, format="mp4"),
                success,
                epoch_idx,
                np.sum(rewards),
                episode_return,
                n_steps,
            ]
        )

        if rollout_save_dir is not None and (save_failures or success):
            # Save the raw rollout data
            save_raw_rollout(
                robot_states=robot_states[: n_steps + 1],
                imgs1=video1[: n_steps + 1],
                imgs2=video2[: n_steps + 1],
                parts_poses=parts_poses[: n_steps + 1],
                actions=actions[:n_steps],
                rewards=rewards[:n_steps],
                success=success,
                furniture=furniture,
                action_type=env.action_type,
                rollout_save_dir=rollout_save_dir,
                compress_pickles=compress_pickles,
            )

    # Sort the table rows by return (highest at the top)
    table_rows = sorted(table_rows, key=lambda x: x[4], reverse=True)

    for row in table_rows:
        tbl.add_data(*row)

    # Log the videos to wandb table if a run is active
    if wandb.run is not None:
        wandb.log(
            {
                "rollouts": tbl,
                "epoch": epoch_idx,
                "epoch_mean_return": total_return / n_rollouts,
            }
        )
        print("logging wandb!")

    pbar.close()

    return RolloutStats(
        success_rate=n_success / n_rollouts,
        n_success=n_success,
        n_rollouts=n_rollouts,
        epoch_idx=epoch_idx,
        rollout_max_steps=rollout_max_steps,
        total_return=total_return,
        total_reward=total_reward,
    )


def do_rollout_evaluation(
    config: DictConfig,
    env: FurnitureSimEnv,
    save_rollouts: bool,
    actor: Actor,
    best_success_rate: float,
    epoch_idx: int,
) -> float:
    rollout_save_dir = None

    if save_rollouts:
        rollout_save_dir = trajectory_save_dir(
            environment="sim",
            task=env.furniture_name,
            demo_source="rollout",
            randomness=config.randomness,
            # Don't create here because we have to do it when we save anyway
            create=False,
        )

    actor.set_task(furniture2idx[env.furniture_name])

    rollout_stats = calculate_success_rate(
        env,
        actor,
        n_rollouts=config.rollout.count,
        rollout_max_steps=config.rollout.max_steps,
        epoch_idx=epoch_idx,
        gamma=config.discount,
        rollout_save_dir=rollout_save_dir,
        save_failures=config.rollout.save_failures,
    )
    success_rate = rollout_stats.success_rate
    best_success_rate = max(best_success_rate, success_rate)

    # Log the success rate to wandb
    wandb.log(
        {
            "success_rate": success_rate,
            "best_success_rate": best_success_rate,
            "n_success": rollout_stats.n_success,
            "n_rollouts": rollout_stats.n_rollouts,
            "epoch": epoch_idx,
        }
    )

    return best_success_rate

import os
import argparse
import os.path as osp

import furniture_bench
from furniture_bench.device import make_device
from furniture_bench.data.data_collector_sm import DataCollectorSpaceMouse
from furniture_bench.config import config
from furniture_bench.envs.initialization_mode import Randomness


def main():
    parser = argparse.ArgumentParser(description="Collect IL data")
    parser.add_argument(
        "--out-data-path", help="Path to directory to save the data", required=True
    )
    parser.add_argument(
        "--furniture",
        help="Name of the furniture",
        choices=list(config["furniture"].keys()),
        required=True,
    )
    parser.add_argument(
        "--is-sim",
        action="store_true",
        help="Use simulator, else use real world environment.",
    )
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Use scripted function for getting action.",
    )
    parser.add_argument(
        "--pkl-only",
        action="store_true",
        help="Only save the pickle file, not .mp4 and .pngs",
    )
    parser.add_argument(
        "--save-failure",
        action="store_true",
        help="Save failure trajectories.",
    )
    parser.add_argument(
        "--headless", help="With front camera view", action="store_true"
    )
    parser.add_argument(
        "--draw-marker", action="store_true", help="Draw AprilTag marker"
    )
    parser.add_argument(
        "--manual-label",
        action="store_true",
        help="Manually label the reward",
    )
    parser.add_argument("--randomness", default="low", choices=["low", "med", "high"])
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--num-demos", default=100, type=int)

    parser.add_argument(
        "--ctrl-mode",
        type=str,
        help="Type of low level controller to use.",
        choices=["osc", "diffik"],
        default="osc",
    )

    parser.add_argument(
        "--no-ee-laser",
        action="store_false",
        help="If set, will not show the laser coming from the end effector",
        dest="ee_laser",
    )

    args = parser.parse_args()

    keyboard_device_interface = make_device("keyboard")

    data_path = osp.join(args.out_data_path, args.furniture)
    if not osp.isdir(data_path):
        os.makedirs(data_path)

    data_collector = DataCollectorSpaceMouse(
        is_sim=args.is_sim,
        data_path=data_path,
        device_interface=keyboard_device_interface,
        furniture=args.furniture,
        headless=args.headless,
        draw_marker=args.draw_marker,
        manual_label=args.manual_label,
        obs_type='image',
        resize_img_after_sim=False, 
        small_sim_img_size=True,  # raw sim images come downsized (i.e., don't call separate resize function)
        scripted=args.scripted,
        randomness=args.randomness,
        gpu_id=args.gpu_id,
        pkl_only=args.pkl_only,
        save_failure=args.save_failure,
        num_demos=args.num_demos,
        ctrl_mode=args.ctrl_mode,
        ee_laser=args.ee_laser
    )
    data_collector.collect()


if __name__ == "__main__":
    main()

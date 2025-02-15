from src.dataset.dataset import (
    FurnitureImageDataset,
)
from src.common.files import get_processed_paths
from omegaconf import DictConfig, OmegaConf
import hydra
from src.dataset import get_normalizer
from src.dataset.normalizer import Normalizer
from PIL import Image
from torchvision.transforms import ToPILImage

OmegaConf.register_new_resolver("eval", eval)

def to_native(obj):
    try:
        return OmegaConf.to_object(obj)
    except ValueError:
        return obj

def set_dryrun_params(config: DictConfig):
    if config.dryrun:
        OmegaConf.set_struct(config, False)
        config.training.steps_per_epoch = 1
        config.data.data_subset = 1

        if config.rollout.rollouts:
            config.rollout.every = 1
            config.rollout.num_rollouts = 1
            config.rollout.loss_threshold = float("inf")
            config.rollout.max_steps = 10

        config.wandb.mode = "disabled"

        OmegaConf.set_struct(config, True)

# @hydra.main(config_path="./src/config", config_name="base")
def main():
    # set_dryrun_params(config)
    # OmegaConf.resolve(config)
    data_path = get_processed_paths(
        environment=to_native("sim"),
        task=to_native("one_leg"),
        demo_source=to_native("augmentation"),
        randomness=to_native("low"),
        demo_outcome=to_native("success"),
    )
    normalizer: Normalizer = get_normalizer(
        "min_max", "delta"
    )
    dataset = FurnitureImageDataset(
        dataset_paths=data_path,
        pred_horizon=32,
        obs_horizon=1,
        action_horizon=8,
        normalizer=normalizer.get_copy(),
        augment_image=True,
        data_subset=None,
        control_mode="delta",
        first_action_idx=0,
        pad_after=True,
        max_episode_count=None,
    )

    print(dataset[0]['color_image1'].squeeze(0).shape)

    print(dataset[0].keys())
    # to_pil = ToPILImage()
    # img = to_pil(dataset[0]['color_image1'].squeeze(0).permute(2, 0, 1))
    # img.save('color_image1.png')
    # img = to_pil(dataset[0]['color_image2'].squeeze(0).permute(2, 0, 1))
    # img.save('color_image2.png')
    # if dataset[0]['depth_image1'] is not None:
    #     img = to_pil(dataset[0]['depth_image1'].squeeze(0).permute(2, 0, 1))
    #     img.save('depth_image1.png')
    #     img = to_pil(dataset[0]['depth_image2'].squeeze(0).permute(2, 0, 1))
    #     img.save('depth_image2.png')
    # print(len(dataset))
    # print(dataset[0]['parts_poses'].shape)
    # print(dataset[0]['parts_poses'])

if __name__ == "__main__":  
    main()
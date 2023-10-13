"""
Triplane DataManager
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset

class MipDataset(InputDataset):
    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["lossmult"]

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1):
        super().__init__(dataparser_outputs, scale_factor)
        if "lossmult" in self.metadata:
            self.lossmult = torch.Tensor(self.metadata["lossmult"])
        else:
            self.lossmult = None

    def get_metadata(self, data: Dict) -> Dict:

        return super().get_metadata(data)

@dataclass
class TriplaneDataManagerConfig(VanillaDataManagerConfig):
    """Triplane DataManager Config

    Add your custom datamanager config parameters here.
    """

    _target: Type = field(default_factory=lambda: TriplaneDataManager)


class TriplaneDataManager(VanillaDataManager[MipDataset]):
    """Triplane DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: TriplaneDataManagerConfig

    def __init__(
        self,
        config: TriplaneDataManagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, **kwargs
        )

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        image_batch = next(self.iter_train_image_dataloader)

        assert self.train_pixel_sampler is not None
        assert isinstance(image_batch, dict)
        batch = self.train_pixel_sampler.sample(image_batch)
        ray_indices = batch["indices"]
        if self.train_dataset.lossmult is not None:
            batch["lossmult"] = self.train_dataset.lossmult[ray_indices[..., 0]]
        ray_bundle = self.train_ray_generator(ray_indices)
        return ray_bundle, batch


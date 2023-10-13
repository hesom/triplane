# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for multiscale blender dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json


@dataclass
class MipBlenderDataParserConfig(DataParserConfig):
    """Blender dataset parser config"""

    _target: Type = field(default_factory=lambda: MipBlender)
    """target class to instantiate"""
    data: Path = Path("data/blender/lego")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""


@dataclass
class MipBlender(DataParser):
    """Blender Dataset
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
    """

    config: MipBlenderDataParserConfig 

    def __init__(self, config: MipBlenderDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color

    def _generate_dataparser_outputs(self, split="train"):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None

        meta = load_from_json(self.data / f"metadata.json")[split]
        image_filenames = []
        poses = []
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        lossmult = []

        for p in meta["file_path"]:
            image_filenames.append(self.data / Path(p))

        for pose in meta["cam2world"]:
            poses.append(np.array(pose))

        for h in meta["height"]:
            height.append(int(h))
            cy.append(h / 2.0)

        for w in meta["width"]:
            width.append(int(w))
            cx.append(w / 2.0)
        
        for f in meta["focal"]:
            fx.append(float(f))
            fy.append(float(f))

        for lm in meta["lossmult"]:
            lossmult.append(float(lm))

        poses = torch.from_numpy(np.array(poses).astype(np.float32)[:, :3])
        fx = torch.tensor(fx, dtype=torch.float32)
        fy = torch.tensor(fy, dtype=torch.float32)
        cx = torch.tensor(cx, dtype=torch.float32)
        cy = torch.tensor(cy, dtype=torch.float32)
        scene_box = SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32))

        cameras = Cameras(
            camera_to_worlds=poses,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            metadata={
                "lossmult": lossmult
            },
        )

        return dataparser_outputs

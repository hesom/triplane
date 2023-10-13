"""
Nerfstudio Triplane Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import (
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

from triplane.triplane_datamanager import (
    TriplaneDataManagerConfig,
)
from triplane.triplane_model import TriplaneModelConfig
from triplane.triplane_pipeline import (
    TriplanePipelineConfig,
)
from triplane.triplane_dataparser import MipBlenderDataParserConfig 

triplane = MethodSpecification(
    config=TrainerConfig(
        method_name="triplane",
        steps_per_eval_batch=500,
        steps_per_save=2000,
        steps_per_eval_all_images=30000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=TriplanePipelineConfig(
            datamanager=TriplaneDataManagerConfig(
                dataparser=MipBlenderDataParserConfig(),
                train_num_rays_per_batch=4096,
                eval_num_rays_per_batch=4096,
            ),
            model=TriplaneModelConfig(regularization="none",),
        ),
        optimizers={
            "fields": {
                "optimizer": AdamOptimizerConfig(lr=0.001),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=30000),
            },
            "encodings": {
                "optimizer": AdamOptimizerConfig(lr=0.02),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.002, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Nerfstudio Triplane.",
)

"""
Triplane Model File

Currently this subclasses the Nerfacto model. Consider subclassing from the base Model.
        if step < self.upsampling_iters[0]:
            return

        new_iters = list(self.upsampling_iters) + [step + 1]
        new_iters.sort()

        index = new_iters.index(step + 1)
        new_grid_resolution = self.upsampling_steps[index - 1]

        self.field.density_encoding.upsample_grid(new_grid_resolution)
        self.field.color_encoding.upsample_grid(new_grid_resolution)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from typing import Dict, Union, List, Literal, Tuple, Type, cast
import math

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from triplane.triplane_field import TriplaneField
from nerfstudio.model_components.losses import MSELoss, tv_loss
from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import AABBBoxCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, misc

@dataclass
class TriplaneModelConfig(ModelConfig):
    """Triplane model config"""

    _target: Type = field(default_factory=lambda: TriplaneModel)
    """target class to instantiate"""
    triplane_resolution: int = 128
    """final render resolution"""
    loss_coefficients: Dict[str, float] = to_immutable_dict(
        {
            "rgb_loss": 1.0,
            "tv_reg_density": 1e-3,
            "tv_reg_color": 1e-4,
            "l1_reg": 5e-4,
        }
    )
    """Loss specific weights."""
    num_samples: int = 50
    """Number of samples in field evaluation"""
    num_uniform_samples: int = 200
    """Number of samples in density evaluation"""
    num_den_components: int = 16
    """Number of components in density encoding"""
    num_color_components: int = 48
    """Number of components in color encoding"""
    appearance_dim: int = 27
    """Number of channels for color encoding"""
    regularization: Literal["none", "l1", "tv"] = "l1"
    """Regularization method used in tensorf paper"""
    background_color: Literal["random", "last_sample", "black", "white"] = "white"
    """Whether to randomize the background color"""
    mip_levels: int = 3
    """Number of triplane mip levels"""
    mip_method: Literal["mip", "laplace", "explicit"] = "mip"
    """Mip method"""
    triplane_reduce: Literal["sum", "product"] = "product"
    """Triplane reduction method"""

class TriplaneModel(Model):
    """Triplane Model

    Args:
        config: Triplane configuration to instantiate model
    """

    config: TriplaneModelConfig

    def __init__(
        self,
        config: TriplaneModelConfig,
        **kwargs,
    ) -> None:
        self.triplane_resolution = config.triplane_resolution
        self.num_den_components = config.num_den_components
        self.num_color_components = config.num_color_components
        self.appearance_dim = config.appearance_dim
        self.triplane_reduce = config.triplane_reduce
        self.mip_levels = config.mip_levels
        self.mip_method = config.mip_method
        super().__init__(config=config, **kwargs)

    def get_training_callbacks(self, training_callback_attributes: TrainingCallbackAttributes) -> List[TrainingCallback]:
        callbacks = []

        return callbacks

    def update_to_step(self, step: int) -> None:
        return

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        texel_base_size = self.scene_box.get_diagonal_length().item() / math.sqrt(2)
        texel_base_size /= self.triplane_resolution

        self.field = TriplaneField(
            self.scene_box.aabb,
            num_den_components=self.num_den_components,
            num_color_components=self.num_color_components,
            init_resolution=self.triplane_resolution,
            appearance_dim=self.appearance_dim,
            head_mlp_num_layers=2,
            head_mlp_layer_width=128,
            scene_scale=15.0,
            texel_base_size=texel_base_size,
            mip_method=self.mip_method, # type: ignore
            mip_levels=self.mip_levels,
            triplane_reduce=self.triplane_reduce # type: ignore
        )

        # samplers
        self.sampler_uniform = UniformSampler(num_samples=self.config.num_uniform_samples, single_jitter=True)
        self.sampler_pdf = PDFSampler(num_samples=self.config.num_samples, single_jitter=True, include_original=False)
        
        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss(reduction="mean")

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # colliders
        if self.config.enable_collider:
            self.collider = AABBBoxCollider(scene_box=self.scene_box)
        
        # regularizations

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}

        param_groups["fields"] = (
            list(self.field.mlp_head.parameters())
            + list(self.field.B.parameters())
            + list(self.field.field_output_rgb.parameters())
        )
        param_groups["encodings"] = list(self.field.color_encoding.parameters()) + list(
                self.field.density_encoding.parameters()
        )

        return param_groups

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, Union[torch.Tensor, List]]:
        # uniform sampling
        ray_samples_uniform = self.sampler_uniform(ray_bundle)
        dens = self.field.get_density(ray_samples_uniform)
        weights = ray_samples_uniform.get_weights(dens)
        coarse_accumulation = self.renderer_accumulation(weights)
        acc_mask = torch.where(coarse_accumulation < 0.0001, False, True).reshape(-1)

        # pdf sampling
        ray_samples_pdf = self.sampler_pdf(ray_bundle, ray_samples_uniform, weights)

        # fine field
        field_outputs_fine = self.field.forward(
            ray_samples_pdf, mask=acc_mask, bg_color=colors.WHITE.to(weights.device)
        )

        mip_selector = self.field.get_mip_selector(ray_samples_uniform)
        mip_selector = torch.median(mip_selector, dim=1)[0]

        weights_fine = ray_samples_pdf.get_weights(field_outputs_fine[FieldHeadNames.DENSITY])
        
        accumulation = self.renderer_accumulation(weights_fine)
        depth = self.renderer_depth(weights_fine, ray_samples_pdf)

        rgb = self.renderer_rgb(
            rgb=field_outputs_fine[FieldHeadNames.RGB],
            weights=weights_fine,
        )
        
        rgb = torch.where(accumulation < 0, colors.WHITE.to(rgb.device), rgb)
        accumulation = torch.clamp(accumulation, min=0)

        outputs = {"rgb": rgb, "accumulation": accumulation, "depth": depth, "mip": mip_selector}
        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses
        device = outputs["rgb"].device
        image = batch["image"][..., :3].to(device)
        lossmult = batch["lossmult"].to(device)
        pred_image, image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image
        )

        rgb_loss = torch.mean(lossmult * self.rgb_loss(image, pred_image))

        loss_dict = {"rgb_loss": rgb_loss}

        if self.config.regularization == "l1":
            l1_parameters = []
            for parameter in self.field.density_encoding.parameters():
                l1_parameters.append(parameter.view(-1))
            loss_dict["l1_reg"] = torch.abs(torch.cat(l1_parameters)).mean()
        elif self.config.regularization == "tv":
            density_plane_coef = self.field.density_encoding.mip_maps
            color_plane_coef = self.field.color_encoding.mip_maps
            assert isinstance(color_plane_coef, torch.Tensor) and isinstance(
                density_plane_coef, torch.Tensor), "TV reg only supported for TensoRF encoding types with plane_coef attribute"
            loss_dict["tv_reg_density"] = tv_loss(density_plane_coef)
            loss_dict["tv_reg_color"] = tv_loss(color_plane_coef)
        elif self.config.regularization == "none":
            pass
        else:
            raise ValueError(f"Regularization {self.config.regularization} not supported")

        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        image = self.renderer_rgb.blend_background(image)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        assert self.config.collider_params is not None
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = cast(torch.Tensor, self.ssim(image, rgb))
        lpips = self.lpips(image, rgb)

        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim.item()),
            "lpips": float(lpips.item()),
        }
        images_dict = {"img": combined_rgb, "accumulation": acc, "depth": depth}
        return metrics_dict, images_dict

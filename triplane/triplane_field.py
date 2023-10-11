"""
Triplane Nerfstudio Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""

from typing import Literal, Optional, Dict
from jaxtyping import Float, Int, Shaped

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.field_heads import FieldHeadNames, RGBFieldHead
from nerfstudio.field_components.mlp import MLP
from nerfstudio.fields.base_field import Field
from nerfstudio.field_components.encodings import NeRFEncoding

class TriplaneField(Field):
    """Triplane Field"""

    def __init__(
        self,
        aabb: Tensor,
        # the aabb of the dataset
        num_den_components: int = 16,
        # Number of components in density encoding
        num_color_components: int = 48,
        # Number of components in color encoding
        init_resolution: int = 128,
        # Initial render resolution,
        appearance_dim: int = 27,
        # the number of dimensions for the appearance embedding
        head_mlp_num_layers: int = 2,
        # number of layers for the MLP 
        head_mlp_layer_width: int = 128,
        # layer width for the MLP
        scene_scale: float = 1.0
    ) -> None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.scene_scale = scene_scale

        # setting up fields
        self.density_encoding = TriplaneNeRFEncoding(
            resolution=init_resolution,
            num_components=num_den_components,
            mip_levels=3,
            reduce="product",
        )
        self.color_encoding = TriplaneNeRFEncoding(
            resolution=init_resolution,
            num_components=num_color_components,
            mip_levels=3,
            reduce="product"
        )

        self.feature_encoding = NeRFEncoding(in_dim=appearance_dim, num_frequencies=2, min_freq_exp=0, max_freq_exp=2)
        self.direction_encoding = NeRFEncoding(in_dim=3, num_frequencies=2, min_freq_exp=0, max_freq_exp=2)

        self.mlp_head = MLP(
            in_dim=appearance_dim + 3 + self.direction_encoding.get_out_dim() + self.feature_encoding.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            activation=nn.ReLU(),
            out_activation=nn.ReLU(),
        )

        self.B = nn.Linear(in_features=self.color_encoding.get_out_dim(), out_features=appearance_dim, bias=False)

        self.field_output_rgb = RGBFieldHead(in_dim=self.mlp_head.get_out_dim(), activation=nn.Sigmoid())
        
    def get_density(self, ray_samples: RaySamples) -> Tensor:
        positions = ray_samples.frustums.get_positions()
        sample_dist = torch.linalg.norm(positions - ray_samples.frustums.origins, dim=-1, keepdim=True)
        mip_selector = sample_dist / self.scene_scale

        positions = SceneBox.get_normalized_positions(positions, self.aabb)
        positions = positions * 2 - 1
        density = self.density_encoding(torch.cat([positions, mip_selector], dim=-1))
        density_enc = torch.sum(density, dim=-1)[:, :, None]
        relu = nn.ReLU()
        density_enc = relu(density_enc)
        return density_enc

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None) -> Tensor:
        d = ray_samples.frustums.directions
        positions = ray_samples.frustums.get_positions()
        sample_dist = torch.linalg.norm(positions - ray_samples.frustums.origins, dim=-1, keepdim=True)
        mip_selector = sample_dist / self.scene_scale

        positions = SceneBox.get_normalized_positions(positions, self.aabb)
        positions = positions * 2 - 1
        rgb_features = self.color_encoding(torch.cat([positions, mip_selector], dim=-1))
        rgb_features = self.B(rgb_features)

        d_encoded = self.direction_encoding(d)
        rgb_features_encoded = self.feature_encoding(rgb_features)

        out = self.mlp_head(torch.cat([rgb_features, d, rgb_features_encoded, d_encoded], dim=-1))
        rgb = self.field_output_rgb(out)

        return rgb

    def forward(
        self,
        ray_samples: RaySamples,
        compute_normals: bool = False,
        mask: Optional[Tensor] = None,
        bg_color: Optional[Tensor] = None,
    ) -> Dict[FieldHeadNames, Tensor]:
        if compute_normals is True:
            raise ValueError("Surface normals are not currently supported with Triplane")
        if mask is not None and bg_color is not None:
            base_density = torch.zeros(ray_samples.shape)[:, :, None].to(mask.device)
            base_rgb = bg_color.repeat(ray_samples[:, :, None].shape)
            if mask.any():
                input_rays = ray_samples[mask, :]
                density = self.get_density(input_rays)
                rgb = self.get_outputs(input_rays, None)

                base_density[mask] = density
                base_rgb[mask] = rgb

                base_density.requires_grad_()
                base_rgb.requires_grad_()
            density = base_density
            rgb = base_rgb
        else:
            density = self.get_density(ray_samples)
            rgb = self.get_outputs(ray_samples, None)
        
        return {FieldHeadNames.DENSITY: density, FieldHeadNames.RGB: rgb}

class TriplaneNeRFEncoding(Encoding):
    plane_coef: Float[Tensor, "3 num_components resolution resolution"]

    def __init__(
        self,
        resolution: int = 32,
        num_components: int = 64,
        init_scale: float = 0.1,
        mip_levels: int = 1,
        reduce: Literal["sum", "product"] = "product",
    ) -> None:
        super().__init__(in_dim=3)

        self.resolution = resolution
        self.num_components = num_components
        self.init_scale = init_scale
        self.reduce = reduce
        self.mip_levels = mip_levels

        self.mip_maps = nn.ParameterList([
            nn.Parameter(
                self.init_scale * torch.randn((3, self.num_components, self.resolution // i, self.resolution // i))
            )
            for i in range(1, mip_levels + 1)
            ])

    def get_out_dim(self) -> int:
        return self.num_components

    def forward(self, in_tensor: Float[Tensor, "*bs 4"]) -> Float[Tensor, "*bs num_components featuresize"]:
        """Sample features from this encoder. Expects in_tensor to be in range [-1, 1]"""

        original_shape = in_tensor.shape
        in_tensor = in_tensor.reshape(-1, 4)
        plane_coord = in_tensor[..., :3]
        mip_selector = in_tensor[..., 3:]

        level_masks = []
        for i in range(1, self.mip_levels + 1):
            cutoff_low = (i-1)*(1.0 / self.mip_levels)
            cutoff_high = i * (1.0 / self.mip_levels)
            if i == 1:
                level_masks.append(mip_selector <= cutoff_high)
            elif i == self.mip_levels:
                level_masks.append(mip_selector > cutoff_low)
            else:
                level_masks.append(torch.logical_and(mip_selector <= cutoff_high, mip_selector > cutoff_low))

        plane_coord = torch.stack([in_tensor[..., [0,1]], in_tensor[..., [0,2]], in_tensor[..., [1,2]]], dim=0)

        plane_coord = plane_coord.detach().view(3, -1, 1, 2)

        if self.mip_levels == 1:
            plane_features = F.grid_sample(self.mip_maps[0], plane_coord, align_corners=True)
        else:
            plane_features = level_masks[-1].int() * F.grid_sample(self.mip_maps[0], plane_coord, align_corners=True)
            for i, plane_coeff in enumerate(self.mip_maps[1:]):
                plane_features += level_masks[-(i+1)].int() * F.grid_sample(plane_coeff, plane_coord, align_corners=True)

            if self.reduce == "product":
                plane_features = plane_features.prod(0).squeeze(-1).T
            else:
                plane_features = plane_features.sum(0).squeeze(-1).T

        return plane_features.reshape(*original_shape[:-1], self.num_components)

    @torch.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        plane_coef = F.interpolate(
            self.plane_coef.data, size=(resolution, resolution), mode="bilinear", align_corners=True
        )

        self.plane_coef = torch.nn.Parameter(plane_coef)
        self.resolution = resolution

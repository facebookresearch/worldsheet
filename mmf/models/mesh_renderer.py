# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch import nn
import timm.models as models

from mmf.neural_rendering.novel_view_projector import NovelViewProjector
from mmf.neural_rendering.losses import (
    ImageL1Loss, DepthL1Loss, MeshLaplacianLoss, GridOffsetLoss, ZGridL1Loss
)
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.utils.distributed import get_world_size


@registry.register_model("mesh_renderer")
class MeshRenderer(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def config_path(cls):
        return "configs/models/mesh_renderer/defaults.yaml"

    def build(self):
        self.batch_size = self.config.batch_size // get_world_size()
        self.image_size_H = self.config.image_size_H
        self.image_size_W = self.config.image_size_W
        if self.image_size_H != self.image_size_W:
            raise NotImplementedError()
        self.grid_stride = self.config.grid_stride
        if (self.image_size_H % self.grid_stride != 0
                or self.image_size_W % self.grid_stride != 0):
            raise Exception("image sizes must be divisible by grid_stride")
        self.grid_H = self.image_size_H // self.grid_stride + 1
        self.grid_W = self.image_size_W // self.grid_stride + 1

        # Offset and depth prediction
        self.offset_and_depth_predictor = OffsetAndZGridPredictor(
            grid_stride=self.grid_stride,
            grid_H=self.grid_H,
            grid_W=self.grid_W,
            z_min=self.config.z_min,
            z_max=self.config.z_max,
            pred_inv_z=self.config.pred_inv_z,
            backbone_name=self.config.backbone_name,
            backbone_dim=self.config.backbone_dim
        )

        self.novel_view_projector = NovelViewProjector(
            batch_size=self.batch_size,
            image_size_H=self.image_size_H,
            image_size_W=self.image_size_W,
            grid_H=self.grid_H,
            grid_W=self.grid_W,
            blur_radius=self.config.rendering.blur_radius,
            faces_per_pixel=self.config.rendering.faces_per_pixel,
            clip_barycentric_coords=self.config.rendering.clip_barycentric_coords,
            sigma=self.config.rendering.sigma,
            gamma=self.config.rendering.gamma,
            background_color=self.config.rendering.background_color,
            hfov=self.config.rendering.hfov,
            z_background=self.config.rendering.z_background,
            gblur_kernel_size=self.config.rendering.gblur_kernel_size,
            gblur_sigma=self.config.rendering.gblur_sigma,
            gblur_weight_thresh=self.config.rendering.gblur_weight_thresh
        )

        self.build_losses()

    def build_losses(self):
        self.loss_image_l1 = ImageL1Loss()
        self.loss_depth_l1 = DepthL1Loss()
        self.loss_mesh_laplacian = MeshLaplacianLoss(
            self.grid_H * self.grid_W, self.novel_view_projector.faces
        )
        self.loss_grid_offset = GridOffsetLoss(self.grid_H, self.grid_W)
        self.loss_z_grid_l1 = ZGridL1Loss(self.grid_H, self.grid_W)

        self.loss_weights = self.config.loss_weights

    def get_optimizer_parameters(self, config):
        params = [{"params": p for p in self.parameters() if p.requires_grad}]

        return params

    def forward(self, sample_list):
        # use the transformed image (after mean subtraction and normalization) as
        # network input
        xy_offset, z_grid = self.offset_and_depth_predictor(sample_list.trans_img_0)

        # use the original image (RGB value in 0~1) as rendering input
        rendering_results = self.novel_view_projector(
            xy_offset=xy_offset,
            z_grid=z_grid,
            rgb_in=sample_list.orig_img_0,
            R_in=sample_list.R_0,
            T_in=sample_list.T_0,
            R_out_list=[sample_list.R_0, sample_list.R_1],
            T_out_list=[sample_list.T_0, sample_list.T_1]
        )

        losses = self.forward_losses(sample_list, xy_offset, z_grid, rendering_results)
        return {"losses": losses}

    def forward_losses(self, sample_list, xy_offset, z_grid, rendering_results):
        rgba_0_rec, rgba_1_rec = rendering_results["rgba_out_rec_list"]
        depth_0_rec, depth_1_rec = rendering_results["depth_out_rec_list"]
        scaled_verts = rendering_results["scaled_verts"]

        depth_l1_0 = self.loss_depth_l1(
            depth_pred=depth_0_rec, depth_gt=sample_list.depth_0,
            loss_mask=sample_list.depth_mask_0.float()
        )
        depth_l1_1 = self.loss_depth_l1(
            depth_pred=depth_1_rec, depth_gt=sample_list.depth_1,
            loss_mask=sample_list.depth_mask_1.float()
        )
        image_l1_1 = self.loss_image_l1(
            rgb_pred=rgba_1_rec[..., :3], rgb_gt=sample_list.orig_img_1,
            loss_mask=sample_list.depth_mask_1.float()
        )

        z_grid_l1_0 = self.loss_z_grid_l1(
            z_grid_pred=z_grid, depth_gt=sample_list.depth_0,
            depth_loss_mask=sample_list.depth_mask_0.float()
        )

        losses_unscaled = {
            "z_grid_l1_0": z_grid_l1_0,
            "depth_l1_0": depth_l1_0,
            "depth_l1_1": depth_l1_1,
            "image_l1_1": image_l1_1,
            "grid_offset": self.loss_grid_offset(xy_offset),
            "mesh_laplacian": self.loss_mesh_laplacian(scaled_verts),
        }
        for k, v in losses_unscaled.items():
            if not torch.all(torch.isfinite(v)).item():
                raise Exception("loss {} becomes {}".format(k, v.mean().item()))
        losses = {k: (v * self.loss_weights[k]) for k, v in losses_unscaled.items()}

        return losses


class OffsetAndZGridPredictor(nn.Module):
    def __init__(
        self, grid_stride, grid_H, grid_W, z_min, z_max, pred_inv_z, backbone_name,
        backbone_dim
    ):
        super().__init__()

        assert grid_stride % 2 == 0
        self.grid_stride = grid_stride
        self.pad = grid_stride // 2
        self.backbone_dim = backbone_dim
        self.grid_H = grid_H
        self.grid_W = grid_W
        self.z_min = z_min
        self.z_max = z_max
        self.pred_inv_z = pred_inv_z

        network = getattr(models, backbone_name)
        self.backbone = network(pretrained=True, output_stride=grid_stride)
        self.xy_offset_predictor = nn.Conv2d(backbone_dim, 2, kernel_size=1)
        self.z_grid_predictor = nn.Conv2d(backbone_dim, 1, kernel_size=1)
        # allow the vertices to move at most half grid length
        # the relative image width and height are 2 (i.e. -1 to 1)
        # so the grid length x is 2. / (self.grid_W - 1), and similary for y
        xy_offset_scale = torch.tensor(
            [1. / (self.grid_W - 1), 1. / (self.grid_H - 1)], dtype=torch.float32
        )
        self.register_buffer("xy_offset_scale", xy_offset_scale)

    def forward(self, img):
        assert img.size(1) == 3, 'The input image must be in NCHW format.'
        batch_size = img.size(0)
        img_pad = nn.functional.pad(
            img, (self.pad, self.pad, self.pad, self.pad), mode='replicate'
        )
        features = self.backbone.forward_features(img_pad)

        # predict in NCHW and permute NCHW -> NHWC
        xy_offset = self.xy_offset_predictor(features).permute(0, 2, 3, 1)
        xy_offset = torch.tanh(xy_offset)
        z_grid = self.z_grid_predictor(features).permute(0, 2, 3, 1)
        z_grid = torch.sigmoid(z_grid)
        # flip the coordinate axis directions from input image to PyTorch3D screen
        # - input image: x - right, y: down
        # - PyTorch3D screen: x - left, y: up, see https://pytorch3d.org/docs/renderer_getting_started
        # so we flip both horizontally and vertically
        xy_offset = xy_offset.flip([1, 2])
        z_grid = z_grid.flip([1, 2])

        xy_offset = xy_offset * self.xy_offset_scale
        # convert z prediction to the range of (z_min, z_max)
        assert z_grid.size(-1) == 1
        if self.pred_inv_z:
            z_grid = 1. / (z_grid * 0.75 + 0.01) - 1
            z_grid = torch.clamp(z_grid, min=self.z_min, max=self.z_max)
        else:
            z_grid = self.z_min + z_grid * (self.z_max - self.z_min)

        # flatten the prediction to match the mesh vertices
        xy_offset = xy_offset.view(batch_size, self.grid_H * self.grid_W, 2)
        z_grid = z_grid.view(batch_size, self.grid_H * self.grid_W, 1)

        assert torch.all(torch.isfinite(xy_offset)).item()
        assert torch.all(torch.isfinite(z_grid)).item()

        return xy_offset, z_grid

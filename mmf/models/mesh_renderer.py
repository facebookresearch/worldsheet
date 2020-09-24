# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os

import numpy as np
import torch
from torch import nn
import timm.models as models

from mmf.neural_rendering.novel_view_projector import NovelViewProjector
from mmf.neural_rendering.losses import (
    ImageL1Loss, DepthL1Loss, MeshLaplacianLoss, GridOffsetLoss, ZGridL1Loss,
    VGG19PerceptualLoss
)
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.utils.distributed import get_world_size, byte_tensor_to_object


logger = logging.getLogger(__name__)


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
            pred_inv_z_synsin=self.config.pred_inv_z_synsin,
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

        if self.config.save_forward_results:
            os.makedirs(self.config.forward_results_dir, exist_ok=True)

    def build_losses(self):
        self.loss_image_l1 = ImageL1Loss()
        self.loss_depth_l1 = DepthL1Loss()
        self.loss_mesh_laplacian = MeshLaplacianLoss(
            self.grid_H * self.grid_W, self.novel_view_projector.faces,
            use_l2_loss=self.config.mesh_laplacian_use_l2_loss
        )
        self.loss_grid_offset = GridOffsetLoss(self.grid_H, self.grid_W)
        self.loss_z_grid_l1 = ZGridL1Loss(self.grid_H, self.grid_W)
        self.loss_vgg19_perceptual = VGG19PerceptualLoss()

        self.loss_weights = self.config.loss_weights

    def get_optimizer_parameters(self, config):
        backbone_params = [
            p for n, p in self.named_parameters() if p.requires_grad and 'backbone' in n
        ]
        head_params = [
            p for n, p in self.named_parameters() if p.requires_grad and 'backbone' not in n
        ]
        logger.info(
            f"Got {len(backbone_params)} backbone parameters "
            f"and {len(head_params)} head parameters. "
            f"Applying {self.config.backbone_lr} lr on backbone parameters."
        )

        params = [
            {"params": head_params},
            {"params": backbone_params, "lr": self.config.backbone_lr},
        ]

        return params

    def get_offset_and_depth_from_gt(self, sample_list):
        batch_size = sample_list.trans_img_0.size(0)
        device = sample_list.trans_img_0.device
        xy_offset = torch.zeros(batch_size, self.grid_H * self.grid_W, 2, device=device)

        sampling_grid = torch.cat(
            [torch.linspace(1, -1, self.grid_W).view(1, self.grid_W, 1).expand(self.grid_H, -1, 1),  # NoQA
             torch.linspace(1, -1, self.grid_H).view(self.grid_H, 1, 1).expand(-1, self.grid_W, 1)],  # NoQA
            dim=-1
        ).unsqueeze(0).expand(batch_size, -1, -1, -1).to(device)
        # sample ground-truth z-grid from ground-truth depth
        z_grid = nn.functional.grid_sample(
            sample_list.depth_0.unsqueeze(1), sampling_grid, padding_mode="border",
            align_corners=True
        ).view(batch_size, self.grid_H * self.grid_W, 1)

        return xy_offset, z_grid

    def forward(self, sample_list):
        if not self.config.fill_z_with_gt:
            # use the transformed image (after mean subtraction and normalization) as
            # network input
            xy_offset, z_grid = self.offset_and_depth_predictor(sample_list.trans_img_0)
        else:
            xy_offset, z_grid = self.get_offset_and_depth_from_gt(sample_list)

        rendering_results = {}
        if not self.config.train_z_grid_only:
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

        # return only the rendering results and skip loss computation, usually for
        # visualization on-the-fly by calling this model separately (instead of running
        # it within the MMF trainer on MMF datasets)
        if self.config.return_rendering_results_only:
            return rendering_results

        losses = self.forward_losses(sample_list, xy_offset, z_grid, rendering_results)
        if self.config.save_forward_results:
            self.save_forward_results(sample_list, xy_offset, z_grid, rendering_results)
        return {"losses": losses}

    def save_forward_results(self, sample_list, xy_offset, z_grid, rendering_results):
        texture_image_rec = rendering_results["texture_image_rec"]
        rgba_0_rec, rgba_1_rec = rendering_results["rgba_out_rec_list"]
        depth_0_rec, depth_1_rec = rendering_results["depth_out_rec_list"]

        for n_im in range(xy_offset.size(0)):
            image_id = byte_tensor_to_object(sample_list.image_id[n_im])
            save_file = os.path.join(
                self.config.forward_results_dir, '{}_outputs.npz'.format(image_id)
            )
            save_dict = {
                "orig_img_0": sample_list.orig_img_0[n_im],
                "orig_img_1": sample_list.orig_img_1[n_im],
                "depth_0": sample_list.depth_0[n_im],
                "depth_1": sample_list.depth_1[n_im],
                "depth_mask_0": sample_list.depth_mask_0[n_im],
                "depth_mask_1": sample_list.depth_mask_1[n_im],
                "xy_offset": xy_offset[n_im],
                "z_grid": z_grid[n_im],
                "texture_image_rec": texture_image_rec[n_im],
                "rgba_0_rec": rgba_0_rec[n_im],
                "rgba_1_rec": rgba_1_rec[n_im],
                "depth_0_rec": depth_0_rec[n_im],
                "depth_1_rec": depth_1_rec[n_im],
            }
            save_dict = {k: v.detach().cpu().numpy() for k, v in save_dict.items()}
            np.savez(save_file, **save_dict)

    def forward_losses(self, sample_list, xy_offset, z_grid, rendering_results):
        z_grid_l1_0 = self.loss_z_grid_l1(
            z_grid_pred=z_grid, depth_gt=sample_list.depth_0,
            depth_loss_mask=sample_list.depth_mask_0.float()
        )
        losses_unscaled = {
            "z_grid_l1_0": z_grid_l1_0,
            "grid_offset": self.loss_grid_offset(xy_offset),
        }

        if not self.config.train_z_grid_only:
            rgba_0_rec, rgba_1_rec = rendering_results["rgba_out_rec_list"]
            depth_0_rec, depth_1_rec = rendering_results["depth_out_rec_list"]
            scaled_verts = rendering_results["scaled_verts"]
            rgb_1_rec = rgba_1_rec[..., :3]

            depth_l1_0 = self.loss_depth_l1(
                depth_pred=depth_0_rec, depth_gt=sample_list.depth_0,
                loss_mask=sample_list.depth_mask_0.float()
            )
            depth_l1_1 = self.loss_depth_l1(
                depth_pred=depth_1_rec, depth_gt=sample_list.depth_1,
                loss_mask=sample_list.depth_mask_1.float()
            )
            image_l1_1 = self.loss_image_l1(
                rgb_pred=rgb_1_rec, rgb_gt=sample_list.orig_img_1,
                loss_mask=sample_list.depth_mask_1.float()
            )
            if self.loss_weights["vgg19_perceptual_1"] != 0:
                vgg19_perceptual_1 = self.loss_vgg19_perceptual(
                    rgb_pred=rgb_1_rec, rgb_gt=sample_list.orig_img_1,
                    loss_mask=sample_list.depth_mask_1.float()
                )
            else:
                vgg19_perceptual_1 = torch.tensor(0., device=rgb_1_rec.device)

            losses_unscaled.update({
                "depth_l1_0": depth_l1_0,
                "depth_l1_1": depth_l1_1,
                "image_l1_1": image_l1_1,
                "vgg19_perceptual_1": vgg19_perceptual_1,
                "mesh_laplacian": self.loss_mesh_laplacian(scaled_verts),
            })

        for k, v in losses_unscaled.items():
            if not torch.all(torch.isfinite(v)).item():
                raise Exception("loss {} becomes {}".format(k, v.mean().item()))
        losses = {k: (v * self.loss_weights[k]) for k, v in losses_unscaled.items()}

        return losses


class OffsetAndZGridPredictor(nn.Module):
    def __init__(
        self, grid_stride, grid_H, grid_W, z_min, z_max, pred_inv_z, pred_inv_z_synsin,
        backbone_name, backbone_dim
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
        self.pred_inv_z_synsin = pred_inv_z_synsin

        network = getattr(models, backbone_name)
        # the minimum output stride for resnet is 8 pixels
        # if we want lower output stride, add ConvTranspose2d at the end
        resnet_grid_stride = max(grid_stride, 8)
        self.backbone = network(pretrained=True, output_stride=resnet_grid_stride)
        assert resnet_grid_stride % grid_stride == 0
        upsample_stride = resnet_grid_stride // grid_stride
        self.slice_b = upsample_stride // 2
        self.slice_e = upsample_stride - 1 - self.slice_b
        if upsample_stride == 1:
            assert self.slice_b == 0 and self.slice_e == 0
            self.xy_offset_predictor = nn.Conv2d(backbone_dim, 2, kernel_size=1)
            self.z_grid_predictor = nn.Conv2d(backbone_dim, 1, kernel_size=1)
        else:
            # upsample in the final prediction layer
            self.xy_offset_predictor = nn.ConvTranspose2d(
                backbone_dim, 2, kernel_size=upsample_stride, stride=upsample_stride
            )
            self.z_grid_predictor = nn.ConvTranspose2d(
                backbone_dim, 1, kernel_size=upsample_stride, stride=upsample_stride
            )
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
        z_grid = self.z_grid_predictor(features).permute(0, 2, 3, 1)
        # strip boundaries
        xy_offset = slice_output(xy_offset, self.slice_b, self.slice_e)
        z_grid = slice_output(z_grid, self.slice_b, self.slice_e)
        # flip the coordinate axis directions from input image to PyTorch3D screen
        # - input image: x - right, y: down
        # - PyTorch3D screen: x - left, y: up, see https://pytorch3d.org/docs/renderer_getting_started
        # so we flip both horizontally and vertically
        xy_offset = xy_offset.flip([1, 2])
        z_grid = z_grid.flip([1, 2])

        xy_offset = torch.tanh(xy_offset)
        xy_offset = xy_offset * self.xy_offset_scale
        # convert z prediction to the range of (z_min, z_max)
        assert z_grid.size(-1) == 1
        if self.pred_inv_z_synsin:
            z_grid = torch.sigmoid(z_grid - 2.8)
            z_grid = 1. / (z_grid * 10 + 0.01) - 0.1
            z_grid = torch.clamp(z_grid, min=self.z_min, max=self.z_max)
        elif self.pred_inv_z:
            z_grid = torch.sigmoid(z_grid)
            z_grid = 1. / (z_grid * 0.75 + 0.01) - 1
            z_grid = torch.clamp(z_grid, min=self.z_min, max=self.z_max)
        else:
            z_grid = torch.sigmoid(z_grid)
            z_grid = self.z_min + z_grid * (self.z_max - self.z_min)

        # flatten the prediction to match the mesh vertices
        xy_offset = xy_offset.view(batch_size, self.grid_H * self.grid_W, 2)
        z_grid = z_grid.view(batch_size, self.grid_H * self.grid_W, 1)

        assert torch.all(torch.isfinite(xy_offset)).item()
        assert torch.all(torch.isfinite(z_grid)).item()

        return xy_offset, z_grid


def slice_output(nhwc_tensor, slice_b, slice_e):
    if slice_b == 0 and slice_e == 0:
        return nhwc_tensor
    b = slice_b
    e = -slice_e if slice_e > 0 else None
    return nhwc_tensor[:, b:e, b:e]

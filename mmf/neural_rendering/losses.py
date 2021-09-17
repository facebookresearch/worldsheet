# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from collections import defaultdict

import torch
from torch import nn
import torchvision


class ImageL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rgb_pred, rgb_gt, loss_mask=None):
        diff = torch.abs(rgb_pred - rgb_gt)
        if loss_mask is not None:
            diff = diff * loss_mask.unsqueeze(-1)
        # averaging over batch, image height and width, but not channels
        l1_loss = torch.mean(diff) * 3  # multiplying by 3 channels

        return l1_loss


class DepthL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, depth_pred, depth_gt, loss_mask=None):
        diff = torch.abs(depth_pred - depth_gt)
        if loss_mask is not None:
            diff = diff * loss_mask
        # averaging over batch, image height and width
        l1_loss = torch.mean(diff)

        return l1_loss


class ZGridL1Loss(nn.Module):
    def __init__(self, grid_H, grid_W):
        super().__init__()
        self.grid_H = grid_H
        self.grid_W = grid_W

        # flip the coordinate axis directions from input image to PyTorch3D screen
        # - input image: x - right, y: down
        # - PyTorch3D screen: x - left, y: up, see https://pytorch3d.org/docs/renderer_getting_started
        # so we sample from 1 to -1 (instead of typically from -1 to 1)
        sampling_grid = torch.cat(
            [torch.linspace(1, -1, grid_W).view(1, grid_W, 1).expand(grid_H, -1, 1),
             torch.linspace(1, -1, grid_H).view(grid_H, 1, 1).expand(-1, grid_W, 1)],
            dim=-1
        ).unsqueeze(0)
        self.register_buffer("sampling_grid", sampling_grid)

    def forward(self, z_grid_pred, depth_gt, depth_loss_mask):
        sampling_grid = self.sampling_grid.expand(z_grid_pred.size(0), -1, -1, -1)

        # sample ground-truth z-grid from ground-truth depth
        z_grid_gt = nn.functional.grid_sample(
            depth_gt.unsqueeze(1), sampling_grid, padding_mode="border",
            align_corners=True
        ).view(*z_grid_pred.shape)
        z_grid_mask = nn.functional.grid_sample(
            depth_loss_mask.unsqueeze(1), sampling_grid, padding_mode="border",
            align_corners=True
        ).view(*z_grid_pred.shape)

        diff = torch.abs(z_grid_pred - z_grid_gt)
        diff = diff * z_grid_mask

        # averaging over batch, grid height and width
        l1_loss = torch.mean(diff)

        return l1_loss


class MeshLaplacianLoss(nn.Module):
    def __init__(self, num_verts, faces, use_l2_loss):
        super().__init__()

        self.use_l2_loss = use_l2_loss

        edges = set()
        for v1, v2, v3 in faces.tolist():
            edges.update([(v1, v2), (v2, v1), (v1, v3), (v3, v1), (v2, v3), (v3, v2)])

        degrees = defaultdict(int)
        for v1, _ in edges:
            degrees[v1] += 1

        laplacian = torch.zeros(num_verts, num_verts, dtype=torch.float)
        for v1 in range(num_verts):
            for v2 in range(num_verts):
                if v1 == v2:
                    laplacian[v1, v2] = degrees[v1]
                elif (v1, v2) in edges:
                    laplacian[v1, v2] = -1

        # The Laplacian matrix is a difference operator and should sum up to 0
        assert torch.all(torch.sum(laplacian, dim=1).eq(0))

        self.register_buffer("laplacian", laplacian)

    def forward(self, verts):
        assert verts.dim() == 3, 'the verts must be in padded format'
        batch_size = verts.size(0)
        if self.use_l2_loss:
            laplacian_reg = torch.sum(
                torch.square(torch.matmul(self.laplacian, verts))
            ) / batch_size
        else:
            laplacian_reg = torch.sum(
                torch.abs(torch.matmul(self.laplacian, verts))
            ) / batch_size
        return laplacian_reg


class GridOffsetLoss(nn.Module):
    def __init__(self, grid_H, grid_W):
        super().__init__()
        self.grid_H = grid_H
        self.grid_W = grid_W

    def forward(self, xy_offset):
        assert xy_offset.dim() == 3
        batch_size = xy_offset.size(0)
        xy_offset_reg = (
            torch.sum(torch.square(xy_offset[..., 0]) * (self.grid_W - 1)) +
            torch.sum(torch.square(xy_offset[..., 1]) * (self.grid_H - 1))
        ) / batch_size
        return xy_offset_reg


# Modified from SynSin codebase
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(
            pretrained=True
        ).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        # Normalize the image so that it is in the appropriate range
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


# Modified from SynSin codebase
class VGG19PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Set to false so that this part of the network is frozen
        self.model = VGG19(requires_grad=False)
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

        # mean and std from https://pytorch.org/docs/stable/torchvision/models.html
        img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        self.register_buffer("img_mean", img_mean)
        self.register_buffer("img_std", img_std)

    def preprocess_img(self, imgs, loss_mask):
        imgs = (imgs - self.img_mean) / self.img_std
        if loss_mask is not None:
            imgs = imgs * loss_mask.unsqueeze(-1)  # mask invalid image regions
        imgs = imgs.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return imgs

    def forward(self, rgb_pred, rgb_gt, loss_mask=None):
        pred_fs = self.model(self.preprocess_img(rgb_pred, loss_mask))
        gt_fs = self.model(self.preprocess_img(rgb_gt, loss_mask))

        # Collect the losses at multiple layers (need unsqueeze in
        # order to concatenate these together)
        loss = 0
        for i in range(0, len(gt_fs)):
            loss += self.weights[i] * self.criterion(pred_fs[i], gt_fs[i])

        return loss

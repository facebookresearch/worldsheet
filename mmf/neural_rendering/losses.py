from collections import defaultdict

import torch
from torch import nn


class ImageL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, rgb_pred, rgb_gt, loss_mask):
        diff = torch.abs(rgb_pred - rgb_gt)
        diff = diff * loss_mask.unsqueeze(-1)
        # averaging over batch, image height and width, but not channels
        l1_loss = torch.mean(diff) * 3  # multiplying by 3 channels

        return l1_loss


class DepthL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, depth_pred, depth_gt, loss_mask):
        diff = torch.abs(depth_pred - depth_gt)
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
    def __init__(self, num_verts, faces):
        super().__init__()

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
        laplacian_reg = torch.sum(
            torch.square(torch.matmul(self.laplacian, verts))
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

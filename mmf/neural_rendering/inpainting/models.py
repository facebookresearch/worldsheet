# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torch import nn
from .pix2pix_networks import GlobalGenerator, get_norm_layer


class MeshRGBGenerator(nn.Module):
    def __init__(self, G_cfg):
        super().__init__()
        self.G_cfg = G_cfg
        # same as in pix2pixHD
        img_mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        img_std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        self.register_buffer("img_mean", img_mean)
        self.register_buffer("img_std", img_std)

        self.use_alpha_input = G_cfg.use_alpha_input

        self.netG = GlobalGenerator(
            input_nc=(4 if self.use_alpha_input else 3),
            output_nc=3,
            ngf=G_cfg.ngf,
            n_downsampling=G_cfg.n_downsampling,
            n_blocks=G_cfg.n_blocks,
            norm_layer=get_norm_layer(norm_type=G_cfg.norm)
        )

    def forward(self, imgs_in):
        assert imgs_in.size(-1) == 4
        imgs = (imgs_in[..., :3] - self.img_mean) / self.img_std
        if self.use_alpha_input:
            alpha_mask = imgs_in[..., -1].unsqueeze(-1).ge(1e-4).float()
            imgs = torch.cat([imgs, alpha_mask], dim=-1)

        # NHWC -> NCHW  -> NHWC
        # outs is in range -1 to +1 as constrained by Tanh in netG
        outs = self.netG(imgs.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # we output in the original float-point image RGB range 0~1,
        # to be compatible with the mesh rendering results
        # scale the output w/ img_out_scaling on Tanh activation
        imgs_out = outs * self.G_cfg.img_out_scaling + 0.5
        return imgs_out

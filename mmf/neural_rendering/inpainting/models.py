import torch
from torch import nn
from .pix2pix_networks import GlobalGenerator, get_norm_layer


class MeshRGBGenerator(nn.Module):
    def __init__(self, G_cfg):
        super().__init__()
        # mean and std from https://pytorch.org/docs/stable/torchvision/models.html
        # they are different from the (0.5, 0.5, 0.5) mean and std in pix2pixHD
        # but should be fine if used consistently
        img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
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
            alpha = imgs_in[..., -1].unsqueeze(-1)
            alpha_mask = alpha.ge(1e-4).float()
            imgs = torch.cat([imgs, alpha_mask], dim=-1)
        imgs = imgs.permute(0, 3, 1, 2)  # NHWC -> NCHW

        outs = self.netG(imgs)

        outs = outs.permute(0, 2, 3, 1)  # NCHW -> NHWC
        outs = outs * self.img_std + self.img_mean
        return outs

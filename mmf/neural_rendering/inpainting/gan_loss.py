# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import torch
from torch import nn
from .pix2pix_networks import MultiscaleDiscriminator, get_norm_layer
from .synsin_gan import GANLoss


logger = logging.getLogger(__name__)


class MeshGANLosses(nn.Module):
    # Adapted from SynSin's codebase

    def __init__(self, D_cfg):
        super().__init__()
        self.D_cfg = D_cfg
        self.model = MeshRGBDiscriminator(D_cfg)
        self.criterionGAN = GANLoss(gan_mode=D_cfg.gan_mode)
        self.criterionFeat = torch.nn.L1Loss()

        self._trainable_params = [p for p in self.model.parameters() if p.requires_grad]

    def _discriminate(self, fake_image, real_image, alpha_mask):
        # Given fake and real image, return the prediction of discriminator
        # for each fake and real image.
        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_image, real_image], dim=0)
        fake_and_real_mask = torch.cat([alpha_mask, alpha_mask], dim=0)
        discriminator_out = self.model(fake_and_real, fake_and_real_mask)
        pred_fake, pred_real = self._divide_pred(discriminator_out)
        return pred_fake, pred_real

    def _divide_pred(self, pred):
        # Take the prediction of fake and real images from the combined batch
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        assert isinstance(pred, list)
        fake = []
        real = []
        for p in pred:
            assert isinstance(p, list)
            fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
            real.append([tensor[tensor.size(0) // 2:] for tensor in p])

        return fake, real

    def compute_generator_loss(self, fake_img, real_img, alpha_mask):
        pred_fake, pred_real = self._discriminate(fake_img, real_img, alpha_mask)

        g_losses = {}
        g_losses["gan"] = self.criterionGAN(pred_fake, True, for_discriminator=False)
        if not self.D_cfg.no_ganFeat_loss:
            feat_weights = 4.0 / (self.D_cfg.n_layers + 1)
            D_weights = 1.0 / self.D_cfg.num_D
            GAN_Feat_loss = 0
            for i in range(self.D_cfg.num_D):
                assert len(pred_fake[i]) > 1
                for j in range(len(pred_fake[i]) - 1):
                    # for each layer output
                    GAN_Feat_loss += self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach()
                    ) * (self.D_cfg.lambda_feat * feat_weights * D_weights)
            g_losses["gan_feat"] = GAN_Feat_loss

        return g_losses

    def compute_discrimator_loss(self, fake_img, real_img, alpha_mask):
        pred_fake, pred_real = self._discriminate(
            fake_img.detach(), real_img.detach(), alpha_mask.detach()
        )

        d_losses = {
            "d_fake": self.criterionGAN(pred_fake, False, for_discriminator=True),
            "d_real": self.criterionGAN(pred_real, True, for_discriminator=True),
        }
        return d_losses

    def forward(self, fake_img, real_img, alpha_mask, update_discriminator):
        # self._debug_print_param()

        # accumulate discriminator loss' gradients in discriminator parameters
        if update_discriminator:
            self._turn_on_param_grad()
            d_losses = self.compute_discrimator_loss(fake_img, real_img, alpha_mask)
            sum(d_losses.values()).mean().backward()
        else:
            with torch.no_grad():
                d_losses = self.compute_discrimator_loss(fake_img, real_img, alpha_mask)
        # self._debug_log_d_losses(d_losses)

        # we do not want generator loss' gradients to be accumulated in
        # discriminator parameters; turn off their requires_grad flag
        # so that autograd won't populate their gradients
        self._turn_off_param_grad()
        g_losses = self.compute_generator_loss(fake_img, real_img, alpha_mask)
        # also put in the no-gradient version of the discriminator losses
        # so that they can be displayed in the training log
        g_losses.update(
            {f"no_grad_{k}": v.clone().detach() for k, v in d_losses.items()}
        )
        return g_losses

    def _turn_off_param_grad(self):
        for p in self._trainable_params:
            p.requires_grad = False

    def _turn_on_param_grad(self):
        for p in self._trainable_params:
            p.requires_grad = True

    def _debug_print_param(self):
        # print a parameter to see if all GPUs have the same parameters
        # (if they don't, then there is a bug in distributed training)
        from mmf.utils.distributed import get_rank
        if not hasattr(self, '_rank'):
            self._rank = get_rank()
            self._iter = 0
        p = self._trainable_params[0]
        d = p.data.view(-1)[:20]
        print(f"iter: {self._iter}, rank: {self._rank}, param: {str(d)}", force=True)
        self._iter += 1

    def _debug_log_d_losses(self, d_losses):
        if not hasattr(self, '_print_count'):
            self._print_count = 0
        if self._print_count % 5 == 0 or (not self.training):
            msg = {k: v.item() for k, v in d_losses.items()}
            logger.info(f"{'TRAIN' if self.training else 'EVAL'} d_losses: {msg}")
        self._print_count += 1


class MeshRGBDiscriminator(nn.Module):
    def __init__(self, D_cfg):
        super().__init__()
        # same as in pix2pixHD
        img_mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        img_std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        self.register_buffer("img_mean", img_mean)
        self.register_buffer("img_std", img_std)

        self.use_alpha_input = D_cfg.use_alpha_input

        self.netD = MultiscaleDiscriminator(
            input_nc=(4 if self.use_alpha_input else 3),
            ndf=D_cfg.ndf,
            n_layers=D_cfg.n_layers,
            norm_layer=get_norm_layer(norm_type=D_cfg.norm),
            use_sigmoid=False,  # sigmoid will be added in loss function if needed
            num_D=D_cfg.num_D,
            getIntermFeat=not D_cfg.no_ganFeat_loss
        )

    def forward(self, imgs_in, alpha_mask):
        assert imgs_in.size(-1) == 3
        imgs = (imgs_in - self.img_mean) / self.img_std
        if self.use_alpha_input:
            imgs = torch.cat([imgs, alpha_mask], dim=-1)

        result = self.netD(imgs.permute(0, 3, 1, 2))

        return result

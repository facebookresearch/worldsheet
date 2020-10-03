import logging
import torch
from torch import nn
from .pix2pix_networks import GANLoss, MultiscaleDiscriminator, get_norm_layer


logger = logging.getLogger(__name__)


class MeshGANLosses(nn.Module):
    # Adapted from SynSin's codebase

    def __init__(self, D_cfg):
        super().__init__()
        self.D_cfg = D_cfg
        self.model = MeshRGBDiscriminator(D_cfg)
        self.criterionGAN = GANLoss(use_lsgan=not D_cfg.no_lsgan)
        self.criterionFeat = torch.nn.L1Loss()

        self._trainable_params = [p for p in self.model.parameters() if p.requires_grad]

    def _discriminate(self, fake_image, real_image):
        # Given fake and real image, return the prediction of discriminator
        # for each fake and real image.
        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_image, real_image], dim=0)
        discriminator_out = self.model(fake_and_real)
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

    def compute_generator_loss(self, fake_img, real_img):
        pred_fake, pred_real = self._discriminate(fake_img, real_img)

        g_losses = {}
        g_losses["gan"] = self.criterionGAN(pred_fake, True)
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

    def compute_discrimator_loss(self, fake_img, real_img):
        pred_fake, pred_real = self._discriminate(fake_img.detach(), real_img.detach())

        d_losses = {
            "d_fake": self.criterionGAN(pred_fake, False),
            "d_real": self.criterionGAN(pred_real, True),
        }
        return d_losses

    def forward(self, fake_img, real_img, update_discriminator):
        # self._debug_print_param()

        # accumulate discriminator loss' gradients in discriminator parameters
        if update_discriminator:
            self._turn_on_param_grad()
            d_losses = self.compute_discrimator_loss(fake_img, real_img)
            sum(d_losses.values()).mean().backward()
        else:
            with torch.no_grad():
                d_losses = self.compute_discrimator_loss(fake_img, real_img)

        # we do not want generator loss' gradients to be accumulated in
        # discriminator parameters; turn off their requires_grad flag
        # so that autograd won't populate their gradients
        self._turn_off_param_grad()
        g_losses = self.compute_generator_loss(fake_img, real_img)
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


class MeshRGBDiscriminator(nn.Module):
    def __init__(self, D_cfg):
        super().__init__()
        # same as in pix2pixHD
        img_mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        img_std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        self.register_buffer("img_mean", img_mean)
        self.register_buffer("img_std", img_std)

        self.netD = MultiscaleDiscriminator(
            input_nc=3,
            ndf=D_cfg.ndf,
            n_layers=D_cfg.n_layers,
            norm_layer=get_norm_layer(norm_type=D_cfg.norm),
            use_sigmoid=D_cfg.no_lsgan,
            num_D=D_cfg.num_D,
            getIntermFeat=not D_cfg.no_ganFeat_loss
        )

    def forward(self, imgs_in):
        assert imgs_in.size(-1) == 3
        imgs = (imgs_in - self.img_mean) / self.img_std
        imgs = imgs.permute(0, 3, 1, 2)  # NHWC -> NCHW

        result = self.netD(imgs)

        return result

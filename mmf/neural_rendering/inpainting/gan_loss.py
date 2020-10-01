import logging
from copy import deepcopy
import torch
from torch import nn
from .pix2pix_networks import GANLoss, MultiscaleDiscriminator, get_norm_layer
from mmf.utils.build import build_scheduler


logger = logging.getLogger(__name__)


class MeshGANLosses(nn.Module):
    # Adapted from SynSin's codebase
    # it holds a discriminator model and its optimizer, and implements all the interface
    # for discriminator stepping.

    def __init__(self, D_cfg):
        super().__init__()
        self.D_cfg = D_cfg
        self.model = MeshRGBDiscriminator(D_cfg)
        self.criterionGAN = GANLoss(use_lsgan=not D_cfg.no_lsgan)
        self.criterionFeat = torch.nn.L1Loss()

        # we'll hold off the optimizer (and lr scheduler) initialization
        # until the beginning of the first forward pass or restoration
        # from checkpoint, where the model has been moved to the proper device
        # and wrapped by DistributedDataParallel
        self.is_optimizer_initialized = False
        self.optimizer_state_dict_to_load = None

    def _init_optimizer_and_scheduler(self):
        if self.is_optimizer_initialized:
            return
        self.optimizer_D = torch.optim.Adam(
            list(self.model.parameters()),
            lr=self.D_cfg.optimizer.lr, betas=(self.D_cfg.optimizer.beta1, 0.999)
        )

        self.lr_scheduler_D = None
        if self.D_cfg.use_lr_scheduler:
            self.lr_scheduler_D = build_scheduler(self.optimizer_D, self.D_cfg)

        logger.info("Optimizer and LR Scheduler are initialized in MeshGANLosses")
        self.is_optimizer_initialized = True

        if self.optimizer_state_dict_to_load is not None:
            self._load_optimizer_and_scheduler_state_dict(
                self.optimizer_state_dict_to_load
            )
            self.optimizer_state_dict_to_load = None

    def optimizer_and_scheduler_state_dict(self):
        assert self.is_optimizer_initialized
        state_dict = {"optimizer_D": self.optimizer_D.state_dict()}
        if self.lr_scheduler_D is not None:
            state_dict["lr_scheduler_D"] = self.lr_scheduler_D.state_dict()
        return state_dict

    def load_optimizer_and_scheduler_state_dict(self, state_dict):
        assert self.optimizer_state_dict_to_load is None
        if self.is_optimizer_initialized:
            self._load_optimizer_and_scheduler_state_dict(self, state_dict)
        else:
            # if optimizer has not been initialized yet,
            # hold the state dict and load it later during initialization
            self.optimizer_state_dict_to_load = deepcopy(state_dict)

    def _load_optimizer_and_scheduler_state_dict(self, state_dict):
        self.optimizer_D.load_state_dict(state_dict["optimizer_D"])
        if self.lr_scheduler_D is not None:
            self.lr_scheduler_D.load_state_dict(state_dict["lr_scheduler_D"])

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
        self._init_optimizer_and_scheduler()
        if update_discriminator:
            assert self.training
            # we'll step discriminator first, as it is easier to make
            # discriminator a module
            self.optimizer_D.zero_grad()
            d_losses = self.compute_discrimator_loss(fake_img, real_img)
            sum(d_losses.values()).mean().backward()
            self.optimizer_D.step()
            if self.lr_scheduler_D is not None:
                self.lr_scheduler_D.step()
        else:
            with torch.no_grad():
                d_losses = self.compute_discrimator_loss(fake_img, real_img)

        g_losses = self.compute_generator_loss(fake_img, real_img)
        # also put in the no-gradient version of the discriminator losses
        # so that they can be displayed in the training log
        g_losses.update(
            {f"no_grad_{k}": v.clone().detach() for k, v in d_losses.items()}
        )
        return g_losses


class MeshRGBDiscriminator(nn.Module):
    def __init__(self, D_cfg):
        super().__init__()
        # mean and std from https://pytorch.org/docs/stable/torchvision/models.html
        # they are different from the (0.5, 0.5, 0.5) mean and std in pix2pixHD
        # but should be fine if used consistently
        img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
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

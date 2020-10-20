from torch import nn
from .ssim import ssim
from .perc_sim import PNet


# The PSNR metric
def psnr_metric(img1, img2, mask=None):
    b = img1.size(0)
    if not (mask is None):
        b = img1.size(0)
        mse_err = (img1 - img2).pow(2) * mask
        mse_err = mse_err.reshape(b, -1).sum(dim=1) / (
            3 * mask.reshape(b, -1).sum(dim=1).clamp(min=1)
        )
    else:
        mse_err = (img1 - img2).pow(2).reshape(b, -1).mean(dim=1)

    psnr = 10 * (1 / mse_err).log10()
    return psnr.clamp(max=100)  # clamp following SynSin's eval


# The SSIM metric
def ssim_metric(img1, img2, mask=None):
    return ssim(img1, img2, mask=mask, size_average=False)


# The perceptual similarity metric
def perceptual_sim(img1, img2, vgg16, mask=None):
    if mask is not None:
        img1 = img1 * mask
        img2 = img2 * mask
    # the input image is in range (0, 1), normalize to range (-1, +1)
    dist = vgg16(img1 * 2 - 1, img2 * 2 - 1)
    return dist


class Metrics(nn.Module):
    def __init__(self, metrics_cfg):
        super().__init__()
        self.compute_psnr = metrics_cfg.compute_psnr
        self.compute_ssim = metrics_cfg.compute_ssim
        self.compute_perc_sim = metrics_cfg.compute_perc_sim
        self.uint8_conversion = metrics_cfg.uint8_conversion

        # metrics should not have params that are stored in a model's
        # state dict. For perceptual losses, VGG params are loaded from
        # torchvision.
        assert len(list(self.parameters())) == 0

    def forward(self, rgb_pred, rgb_gt, vis_mask=None):
        # NHWC to NCHW
        assert rgb_pred.size(-1) == 3
        assert rgb_gt.size(-1) == 3
        rgb_pred = rgb_pred.clamp(min=0, max=1)
        if self.uint8_conversion:
            rgb_pred = emulate_uint8_conversion(rgb_pred)
            rgb_gt = emulate_uint8_conversion(rgb_gt)

        rgb_pred = rgb_pred.permute(0, 3, 1, 2)
        rgb_gt = rgb_gt.permute(0, 3, 1, 2)
        if vis_mask is not None:
            assert vis_mask.size(-1) == 1
            vis_mask = vis_mask.permute(0, 3, 1, 2)

        results = {}
        self.add_psnr_results(results, rgb_pred, rgb_gt, vis_mask)
        self.add_ssim_results(results, rgb_pred, rgb_gt, vis_mask)
        self.add_perc_sim_results(results, rgb_pred, rgb_gt, vis_mask)

        # detach and average over batch size
        return results

    def add_psnr_results(self, results, rgb_pred, rgb_gt, vis_mask):
        if not self.compute_psnr:
            return
        results["PSNR"] = psnr_metric(rgb_pred, rgb_gt)
        if vis_mask is not None:
            results["PSNR_Vis"] = psnr_metric(rgb_pred, rgb_gt, vis_mask)
            results["PSNR_InVis"] = psnr_metric(rgb_pred, rgb_gt, 1 - vis_mask)

    def add_ssim_results(self, results, rgb_pred, rgb_gt, vis_mask):
        if not self.compute_ssim:
            return
        results["SSIM"] = ssim_metric(rgb_pred, rgb_gt)
        if vis_mask is not None:
            results["SSIM_Vis"] = ssim_metric(rgb_pred, rgb_gt, vis_mask)
            results["SSIM_InVis"] = ssim_metric(rgb_pred, rgb_gt, 1 - vis_mask)

    def add_perc_sim_results(self, results, rgb_pred, rgb_gt, vis_mask):
        if not self.compute_perc_sim:
            return

        # we'll initialize vgg16 late here, so that we can get the proper device
        # from input, and put it in a list so that it doesn't go into model params
        if not hasattr(self, "vgg16list"):
            vgg16 = PNet().to(rgb_pred.device)
            vgg16.eval()
            for p in vgg16.parameters():
                p.requires_grad = False
            self.vgg16list = [vgg16]

        vgg16 = self.vgg16list[0]
        results["PercSim"] = perceptual_sim(rgb_pred, rgb_gt, vgg16)
        if vis_mask is not None:
            results["PercSim_Vis"] = perceptual_sim(rgb_pred, rgb_gt, vgg16, vis_mask)
            results["PercSim_InVis"] = perceptual_sim(
                rgb_pred, rgb_gt, vgg16, 1 - vis_mask
            )


def emulate_uint8_conversion(im):
    # first convert the image to uint8 and then to float32 again,
    # to emulate the precision loss from saving the image to a uint8 PNG file
    # for offline evaluation
    return (im * 255).round().clamp(min=0, max=255) / 255

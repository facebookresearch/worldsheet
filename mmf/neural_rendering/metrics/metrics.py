from torch import nn
from .ssim import ssim


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


class Metrics(nn.Module):
    def __init__(self, metrics_cfg):
        super().__init__()
        self.compute_psnr = metrics_cfg.compute_psnr
        self.compute_ssim = metrics_cfg.compute_ssim
        self.compute_perc_sim = metrics_cfg.compute_perc_sim

        # metrics parameters shouldn't be optimized
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, rgb_pred, rgb_gt, vis_mask=None):
        # NHWC to NCHW
        assert rgb_pred.size(-1) == 3
        assert rgb_gt.size(-1) == 3
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
        results["psnr"] = psnr_metric(rgb_pred, rgb_gt)
        if vis_mask is not None:
            results["psnr_vis"] = psnr_metric(rgb_pred, rgb_gt, vis_mask)
            results["psnr_invis"] = psnr_metric(rgb_pred, rgb_gt, 1 - vis_mask)

    def add_ssim_results(self, results, rgb_pred, rgb_gt, vis_mask):
        if not self.compute_ssim:
            return
        results["ssim"] = ssim_metric(rgb_pred, rgb_gt)
        if vis_mask is not None:
            results["ssim_vis"] = ssim_metric(rgb_pred, rgb_gt, vis_mask)
            results["ssim_invis"] = ssim_metric(rgb_pred, rgb_gt, 1 - vis_mask)

    def add_perc_sim_results(self, results, rgb_pred, rgb_gt, vis_mask):
        if not self.compute_perc_sim:
            return

        raise NotImplementedError()

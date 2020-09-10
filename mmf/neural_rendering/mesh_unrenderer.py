import numpy as np
import torch
from torch import nn

from pytorch3d.renderer.blending import BlendParams
from pytorch3d.ops import interpolate_face_attributes
from . import softsplat


class MeshUnrenderer(nn.Module):
    def __init__(self, rasterizer, unshader):
        super().__init__()
        self.rasterizer = rasterizer
        self.unshader = unshader

    def forward(self, meshes_world, images, H_out=None, W_out=None, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        textures = self.unshader(fragments, meshes_world, images, H_out, W_out, **kwargs)

        return textures


class SoftPerspectiveUnshader(nn.Module):
    def __init__(
        self, device="cpu", cameras=None, blend_params=None, texture_splatter=None
    ):
        super().__init__()
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()
        self.texture_splatter = texture_splatter

    def forward(
        self, fragments, meshes, images, H_out=None, W_out=None, output_weights=False, **kwargs
    ) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SoftPerspectiveUnshader"
            raise ValueError(msg)
        blend_params = kwargs.get("blend_params", self.blend_params)

        images_unblend, ones_unblind = softmax_rgb_unblend(
            images, fragments, blend_params
        )
        textures, texture_weights = meshes_unsample_textures(
            fragments, meshes, images_unblend, ones_unblind, H_out, W_out,
            output_weights, self.texture_splatter, **kwargs
        )
        return textures, texture_weights


def softmax_rgb_unblend(
    images, fragments, blend_params, znear: float = 1.0, zfar: float = 100
):
    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device

    # Background color
    delta = np.exp(1e-10 / blend_params.gamma) * 1e-10
    delta = torch.tensor(delta, device=device)

    # Mask for padded pixels.
    mask = fragments.pix_to_face >= 0

    # Sigmoid probability map based on the distance of the pixel to the face.
    prob_map = torch.sigmoid(-fragments.dists / blend_params.sigma) * mask

    # # The cumulative product ensures that alpha will be 0.0 if at least 1
    # # face fully covers the pixel as for that face, prob will be 1.0.
    # # This results in a multiplication by 0.0 because of the (1.0 - prob)
    # # term. Therefore 1.0 - alpha will be 1.0.
    # alpha = torch.prod((1.0 - prob_map), dim=-1)

    # Weights for each face. Adjust the exponential by the max z to prevent
    # overflow. zbuf shape (N, H, W, K), find max over K.
    # TODO: there may still be some instability in the exponent calculation.

    z_inv = (zfar - fragments.zbuf) / (zfar - znear) * mask
    # pyre-fixme[16]: `Tuple` has no attribute `values`.
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
    z_inv_max = torch.max(z_inv, dim=-1).values[..., None]
    # pyre-fixme[6]: Expected `Tensor` for 1st param but got `float`.
    weights_num = prob_map * torch.exp((z_inv - z_inv_max) / blend_params.gamma)

    # Normalize weights.
    # weights_num shape: (N, H, W, K). Sum over K and divide through by the sum.
    denom = weights_num.sum(dim=-1)[..., None] + delta
    weights = weights_num / denom

    # images_unblend: (N, H, W, K, C)
    images_unblend = images.unsqueeze(-2) * weights.unsqueeze(-1)

    ones_unblind = weights.unsqueeze(-1).expand(-1, -1, -1, -1, 3)
    return images_unblend, ones_unblind


def meshes_unsample_textures(
    fragments, meshes, images_unblend, ones_unblind, H_out=None, W_out=None,
    output_weights=False, texture_splatter=None, **kwargs
):
    texture = meshes.textures
    if texture.isempty():
        faces_verts_uvs = torch.zeros(
            (texture._N, 3, 2), dtype=torch.float32, device=texture.device
        )
    else:
        packing_list = [
            i[j] for i, j in zip(texture.verts_uvs_list(), texture.faces_uvs_list())
        ]
        faces_verts_uvs = torch.cat(packing_list)

    # pixel_uvs: (N, H, W, K, 2)
    pixel_uvs = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts_uvs
    )

    N, H, W, K = fragments.pix_to_face.shape
    N_i, H_i, W_i, K_i, C = images_unblend.shape
    assert N_i == N and H_i == H and W_i == W and K_i == K

    if H_out is None:
        H_out = H
    if W_out is None:
        W_out = W

    # pixel_uvs: (N, H, W, K, 2) -> (N, K, H, W, 2) -> (NK, H, W, 2)
    pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(N * K, H, W, 2)

    # images_unblend: (N, H, W, K, C) -> (N, K, C, H, W) -> (N*K, C, H, W)
    images_unblend = images_unblend.permute(0, 3, 4, 1, 2).reshape(N * K, C, H, W)

    # ones_unblind: (N, H, W, K, C) -> (N, K, C, H, W) -> (N*K, C, H, W)
    ones_unblind = ones_unblind.permute(0, 3, 4, 1, 2).reshape(N * K, C, H, W)

    pixel_uvs = pixel_uvs * 2.0 - 1.0

    # uv_texels, uv_ones: (N*K, C, H, W) -> (N, K, C, H, W)
    uv_texels = grid_unsample_no_normalization(
        images_unblend, pixel_uvs, H_out, W_out
    )
    uv_texels = uv_texels.reshape(N, K, C, H_out, W_out)
    uv_ones = grid_unsample_no_normalization(
        ones_unblind, pixel_uvs, H_out, W_out
    )
    uv_ones = uv_ones.reshape(N, K, C, H_out, W_out)

    # textures, weights: (N, C, H, W) -> (N, H, W, C)
    textures = torch.sum(uv_texels, dim=1)
    weights = torch.sum(uv_ones, dim=1)
    textures = textures / torch.clamp(weights, min=1e-8)
    if texture_splatter is not None:
        textures = texture_splatter(textures, weights)
    textures = textures.reshape(N, C, H_out, W_out).permute(0, 2, 3, 1)
    textures = torch.flip(textures, [1])  # flip y axis of the images

    if output_weights:
        texture_weights = weights.reshape(N, C, H_out, W_out).permute(0, 2, 3, 1)
        texture_weights = torch.flip(texture_weights, [1])  # flip y axis of the images
    else:
        texture_weights = None

    return textures, texture_weights


def _get_identity(H, W, device):
    identity = torch.cat([
        torch.linspace(0, W, W, device=device).view(1, 1, W, 1).expand(1, H, W, 1),
        torch.linspace(0, H, H, device=device).view(1, H, 1, 1).expand(1, H, W, 1),
    ], dim=-1)

    return identity


def grid_unsample_no_normalization(image, grid, H_out=None, W_out=None):
    _, H, W, _ = grid.shape
    _, _, H_im, W_im = image.shape
    assert H_im == H and W_im == W

    if H_out is None:
        H_out = H
    if W_out is None:
        W_out = W
    if H_out > H or W_out > W:
        raise NotImplementedError()

    identity = _get_identity(H, W, grid.device)
    out_scale = torch.tensor(
        [W_out, H_out], dtype=torch.float, device=image.device
    )
    flow = (grid / 2. + .5) * out_scale - identity
    flow = flow.permute(0, 3, 1, 2)

    image = image.contiguous()
    flow = flow.contiguous()
    output = softsplat.FunctionSoftsplat(
        tenInput=image, tenFlow=flow, tenMetric=None, strType='summation'
    )

    if H_out != H or W_out != W:
        output = output[:, :, :H_out, :W_out]

    return output


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
    """
    def __init__(self, channels, kernel_size, sigma):
        super().__init__()

        # 1D Gaussian kernel
        grid = torch.arange(kernel_size, dtype=torch.float32)
        mu = (kernel_size - 1) / 2
        kernel = torch.exp((-((grid - mu) / sigma) ** 2) / 2.)
        kernel /= torch.sum(kernel)

        # separate into two convolutions to save computation
        # vertical conv
        self.conv_w = torch.nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
            groups=channels,
            bias=False,
            padding_mode='replicate',
        )
        self.conv_w.weight.requires_grad = False
        self.conv_w.weight[...] = kernel.view(1, kernel_size)

        # horizontal conv
        self.conv_h = torch.nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(kernel_size, 1),
            padding=(kernel_size // 2, 0),
            groups=channels,
            bias=False,
            padding_mode='replicate',
        )
        self.conv_h.weight.requires_grad = False
        self.conv_h.weight[...] = kernel.view(kernel_size, 1)

    def forward(self, images, nhwc_input=False):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        if nhwc_input:
            images = images.permute(0, 3, 1, 2)

        out = self.conv_h(self.conv_w(images))

        if nhwc_input:
            out = out.permute(0, 2, 3, 1)
        return out


class GaussianSplatter(nn.Module):
    def __init__(self, channels, kernel_size, sigma, weight_thresh=1e-4):
        super().__init__()

        self.smoothing = GaussianSmoothing(channels, kernel_size, sigma)
        self.weight_thresh = weight_thresh

    def forward(self, images, weights, nhwc_input=False):
        if nhwc_input:
            images = images.permute(0, 3, 1, 2)
            weights = weights.permute(0, 3, 1, 2)

        mask = weights.gt(self.weight_thresh).float()
        image_blur = self.smoothing(images * mask)
        mask_blur = self.smoothing(mask)
        image_blur_norm = image_blur / torch.clamp(mask_blur, min=1e-8)

        out = images * mask + image_blur_norm * (1 - mask)

        if nhwc_input:
            out = out.permute(0, 2, 3, 1)
        return out

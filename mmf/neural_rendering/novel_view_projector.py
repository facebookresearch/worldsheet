import logging

import numpy as np
import torch
from torch import nn
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    TexturesUV, FoVPerspectiveCameras, RasterizationSettings, MeshRenderer,
    MeshRasterizer, PointLights, HardFlatShader
)
from pytorch3d.renderer.blending import BlendParams

from .perspective_shader import SoftRGBDShader
from .mesh_unrenderer import (
    MeshUnrenderer, SoftPerspectiveUnshader, GaussianSplatter
)


logger = logging.getLogger(__name__)


class NovelViewProjector(nn.Module):
    def __init__(
        self,
        batch_size,
        image_size_H,
        image_size_W,
        grid_H,
        grid_W,
        blur_radius,
        faces_per_pixel,
        clip_barycentric_coords,
        sigma,
        gamma,
        background_color,
        hfov,
        z_background,
        gblur_kernel_size,
        gblur_sigma,
        gblur_weight_thresh,
        directly_use_img_as_texture
    ):
        super().__init__()

        self.batch_size = batch_size

        # Rendering
        self.raster_settings = RasterizationSettings(
            image_size=image_size_H,
            blur_radius=blur_radius,
            faces_per_pixel=faces_per_pixel,
            perspective_correct=True,
            clip_barycentric_coords=clip_barycentric_coords,
        )
        self.blend_params = BlendParams(
            sigma=sigma,
            gamma=gamma,
            background_color=background_color
        )
        self.cameras = FoVPerspectiveCameras(fov=hfov)
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizerWithZClamp(
                cameras=self.cameras,
                raster_settings=self.raster_settings
            ),
            shader=SoftRGBDShader(
                cameras=self.cameras,
                blend_params=self.blend_params,
                z_background=z_background
            )
        )
        self.unrenderer = MeshUnrenderer(
            rasterizer=MeshRasterizerWithZClamp(
                cameras=self.cameras,
                raster_settings=self.raster_settings
            ),
            unshader=SoftPerspectiveUnshader(
                cameras=self.cameras,
                blend_params=self.blend_params,
                texture_splatter=GaussianSplatter(
                    channels=3,
                    kernel_size=gblur_kernel_size,
                    sigma=gblur_sigma,
                    weight_thresh=gblur_weight_thresh
                )
            )
        )

        verts = torch.zeros((grid_H * grid_W, 3), dtype=torch.float)
        self.verts = verts
        # a grid mesh

        def _idx(yy, xx):
            return yy * grid_W + xx

        def _reverse(F, reverse):
            return F[::-1] if reverse else F

        flip = False
        faces = torch.tensor(
            [_reverse([_idx(ny, nx + 1), _idx(ny, nx), _idx(ny + 1, nx)], flip)
             for ny in range(0, grid_H - 1, 2) for nx in range(1, grid_W - 1, 2)] +
            [_reverse([_idx(ny, nx + 1), _idx(ny + 1, nx), _idx(ny + 1, nx + 1)], flip)
             for ny in range(0, grid_H - 1, 2) for nx in range(1, grid_W - 1, 2)] +
            [_reverse([_idx(ny, nx + 1), _idx(ny, nx), _idx(ny + 1, nx + 1)], flip)
             for ny in range(0, grid_H - 1, 2) for nx in range(0, grid_W - 1, 2)] +
            [_reverse([_idx(ny + 1, nx + 1), _idx(ny, nx), _idx(ny + 1, nx)], flip)
             for ny in range(0, grid_H - 1, 2) for nx in range(0, grid_W - 1, 2)] +
            [_reverse([_idx(ny, nx + 1), _idx(ny, nx), _idx(ny + 1, nx)], flip)
             for ny in range(1, grid_H - 1, 2) for nx in range(0, grid_W - 1, 2)] +
            [_reverse([_idx(ny, nx + 1), _idx(ny + 1, nx), _idx(ny + 1, nx + 1)], flip)
             for ny in range(1, grid_H - 1, 2) for nx in range(0, grid_W - 1, 2)] +
            [_reverse([_idx(ny, nx + 1), _idx(ny, nx), _idx(ny + 1, nx + 1)], flip)
             for ny in range(1, grid_H - 1, 2) for nx in range(1, grid_W - 1, 2)] +
            [_reverse([_idx(ny + 1, nx + 1), _idx(ny, nx), _idx(ny + 1, nx)], flip)
             for ny in range(1, grid_H - 1, 2) for nx in range(1, grid_W - 1, 2)],
            dtype=torch.long
        )
        self.faces = faces
        # Texture
        xs_uv = np.linspace(1, 0, grid_W)
        ys_uv = np.linspace(0, 1, grid_H)
        verts_uvs = torch.tensor(
            [[x, y] for y in ys_uv for x in xs_uv],
            dtype=torch.float
        )
        texture_map = 0.5 * torch.ones(
            (image_size_H, image_size_W, 3), dtype=torch.float
        )
        textures = TexturesUV(
            maps=[texture_map.clone().detach() for _ in range(self.batch_size)],
            faces_uvs=[faces.clone().detach() for _ in range(self.batch_size)],
            verts_uvs=[verts_uvs.clone().detach() for _ in range(self.batch_size)]
        )
        self.mesh = Meshes(
            verts=[verts.clone().detach() for _ in range(self.batch_size)],
            faces=[faces.clone().detach() for _ in range(self.batch_size)],
            textures=textures
        )

        xs_vertex = np.linspace(-1, 1, grid_W)
        ys_vertex = np.linspace(-1, 1, grid_H)
        unscaled_grid = torch.tensor(
            [[x, y] for y in ys_vertex for x in xs_vertex],
            dtype=torch.float
        )
        self.register_buffer("unscaled_grid", unscaled_grid)

        x_scale = np.tan(hfov * (np.pi / 180.) / 2.)
        y_scale = np.tan(hfov * (np.pi / 180.) / 2.)
        scaling_factor = torch.tensor([x_scale, y_scale], dtype=torch.float)
        self.register_buffer("scaling_factor", scaling_factor)

        # build a flat shader to render mesh shape only
        from mmf.common.registry import registry
        device = registry.get("current_device")
        self.notexture_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=RasterizationSettings(
                    image_size=image_size_H,
                    blur_radius=1e-12,
                    faces_per_pixel=1,
                    perspective_correct=True,
                    clip_barycentric_coords=clip_barycentric_coords,
                )
            ),
            shader=HardFlatShader(cameras=self.cameras, device=device)
        )
        # a point light at the origin (where the camera is)
        # the light location will be updated during the forward pass
        # to transform the origin (0, 0, 0) to world coordinates
        self.lights = PointLights(
            location=[[0.0, 0.0, 0.0]] * self.batch_size,
            device=device
        )

        self.directly_use_img_as_texture = directly_use_img_as_texture

    def _pad_inputs(self, tensor):
        # when a batch has fewer samples than the typical batch size
        # tile the last example to fill the remaining indices in the batch
        assert tensor.size(0) <= self.batch_size
        if tensor.size(0) < self.batch_size:
            repeats = [self.batch_size - tensor.size(0)] + [1] * (tensor.dim() - 1)
            tensor = torch.cat([tensor, tensor[-1:].repeat(*repeats)], dim=0)
        assert tensor.size(0) == self.batch_size
        return tensor

    def _strip_outputs(self, tensor, actual_batch_size):
        assert tensor.size(0) == self.batch_size
        assert actual_batch_size <= self.batch_size
        if actual_batch_size < self.batch_size:
            tensor = tensor[:actual_batch_size]
        return tensor

    def forward(
        self, xy_offset, z_grid, rgb_in, R_in, T_in, R_out_list, T_out_list,
        render_mesh_shape=False
    ):
        # pad the inputs to batch size
        actual_batch_size = rgb_in.size(0)
        if actual_batch_size != self.batch_size:
            xy_offset = self._pad_inputs(xy_offset)
            z_grid = self._pad_inputs(z_grid)
            rgb_in = self._pad_inputs(rgb_in)
            R_in = self._pad_inputs(R_in)
            T_in = self._pad_inputs(T_in)
            R_out_list = [self._pad_inputs(R_out) for R_out in R_out_list]
            T_out_list = [self._pad_inputs(T_out) for T_out in T_out_list]

        # since "mesh" and "cameras" are not instances of nn.Module,
        # we need to manually move them to another device
        device = xy_offset.device
        if self.mesh.device != device:
            self.mesh = self.mesh.to(device)
            self.cameras = self.cameras.to(device)

        # project the mesh from view 0 screen coordinates to view 0 camera coordinates
        scaled_grid = (self.unscaled_grid + xy_offset) * self.scaling_factor
        ones = torch.ones(
            scaled_grid.shape[:-1] + (1,),
            dtype=torch.float32, device=scaled_grid.device
        )
        scaled_grid3d = torch.cat([scaled_grid, ones], dim=-1)
        scaled_verts = scaled_grid3d * z_grid

        # project from view 0 camera coordinates to world coordinates
        cam2world_in = self.cameras.get_world_to_view_transform(R=R_in, T=T_in).inverse()
        deform_verts = cam2world_in.transform_points(scaled_verts)
        deform_verts = deform_verts.view(-1, 3)  # convert to packed

        # de-rendering to recover the texture map
        deformed_mesh = self.mesh.offset_verts(deform_verts)
        if self.directly_use_img_as_texture:
            # directly using the input image as the mesh texture map
            # this is not perspective-correct
            texture_image_rec = rgb_in
        else:
            # sample the mesh texture through differentiable texture sampler
            # this is perspective-correct (undo barycentric coordinate weighting)
            texture_image_rec, _ = self.unrenderer(
                deformed_mesh, rgb_in, R=R_in, T=T_in,
            )
        deformed_mesh.textures._maps_padded = texture_image_rec

        rgba_out_rec_list = []
        depth_out_rec_list = []
        for R_out, T_out in zip(R_out_list, T_out_list):
            images_out_rec = self.renderer(deformed_mesh, R=R_out, T=T_out)
            assert images_out_rec.size(-1) == 5
            rgba_out_rec = images_out_rec[..., :4]
            depth_out_rec = images_out_rec[..., -1]
            rgba_out_rec_list.append(rgba_out_rec)
            depth_out_rec_list.append(depth_out_rec)

        results = {
            "scaled_verts": scaled_verts,
            "texture_image_rec": texture_image_rec,
            "rgba_out_rec_list": rgba_out_rec_list,
            "depth_out_rec_list": depth_out_rec_list,
        }

        if render_mesh_shape:
            mesh_shape_out_list = []
            for R_out, T_out in zip(R_out_list, T_out_list):
                # transform the camera origin to world coordinates
                # world_lights_location needs to have shape broadcast-able
                # to the (N, H, W, K, 3) shape of pixel normals
                world_lights_location = cam2world_in.transform_points(
                    torch.zeros(self.batch_size, 1, 3, device=R_in.device)
                ).unsqueeze(1).unsqueeze(1)
                self.lights.location = world_lights_location
                deformed_mesh.textures._maps_padded = torch.ones_like(texture_image_rec)
                mesh_shape_out = self.notexture_renderer(
                    deformed_mesh, R=R_out, T=T_out, lights=self.lights
                )
                mesh_shape_out_list.append(mesh_shape_out)

            results["mesh_shape_out_list"] = mesh_shape_out_list

        # strip the outputs to the actual batch size
        if actual_batch_size != self.batch_size:
            for k in results:
                if isinstance(results[k], list):
                    results[k] = [
                        self._strip_outputs(t, actual_batch_size) for t in results[k]
                    ]
                else:
                    assert isinstance(results[k], torch.Tensor)
                    results[k] = self._strip_outputs(results[k], actual_batch_size)

        return results


class MeshRasterizerWithZClamp(MeshRasterizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.z_eps = 1e-2

    def transform(self, meshes_world, **kwargs) -> torch.Tensor:
        # following https://github.com/facebookresearch/pytorch3d/blob/46c0e834616f6950d0ed5a9826a49744d8e83f7e/pytorch3d/renderer/mesh/rasterizer.py#L118
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of MeshRasterizer"
            raise ValueError(msg)

        n_cameras = len(cameras)
        if n_cameras != 1 and n_cameras != len(meshes_world):
            msg = "Wrong number (%r) of cameras for %r meshes"
            raise ValueError(msg % (n_cameras, len(meshes_world)))

        verts_world = meshes_world.verts_padded()

        verts_view = cameras.get_world_to_view_transform(**kwargs).transform_points(
            verts_world
        )
        verts_screen = cameras.get_projection_transform(**kwargs).transform_points(
            verts_view, eps=self.z_eps
        )
        verts_screen[..., 2] = verts_view[..., 2]
        meshes_screen = meshes_world.update_padded(new_verts_padded=verts_screen)
        return meshes_screen

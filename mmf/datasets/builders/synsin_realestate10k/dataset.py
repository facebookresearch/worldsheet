# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch
from mmf.common.sample import Sample
from mmf.datasets.base_dataset import BaseDataset
from mmf.utils.distributed import object_to_byte_tensor
from .realestate10k import RealEstate10K
from .eval_realestate import Dataset as RealEstate10KEval


class SynSinDatasetOption:
    def __init__(self, data_path, video_list, image_size):
        self.W = image_size
        self.train_data_path = data_path
        self.video_list = video_list


class SynSinRealEstate10KDataset(BaseDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__('synsin_realestate10k', config, dataset_type, *args, **kwargs)

        self.num_view_per_sample = config.num_view_per_sample
        if self._dataset_type == "train" or config.save_for_external_inpainting_train:
            opts = SynSinDatasetOption(
                config.train_data_dir, config.train_video_list, config.image_size
            )
            # Always load the training set when `save_for_external_inpainting_train` is
            # on, regardless of `self._dataset_type` (i.e. using the training set images
            # of RealEstate10K for inpainting training)
            # Note that the actual `self._dataset_type` could be `val` or `test`, when
            # `save_for_external_inpainting_train` is used, because in MMF one can only
            # run evaluation or predictions in `val` or `test` mode (but training set
            # images will be actually loaded here even under `val` or `test` mode).
            split_to_load = "train"
            self.synsin_realestate10k = RealEstate10K(
                split_to_load, opts=opts, num_views=self.num_view_per_sample,
                num_duplicate=config.train_num_duplicate
            )
        elif self._dataset_type == "val":
            self.synsin_realestate10k = RealEstate10KEval(
                config.eval_val_frames, config.eval_val_cameras, config.image_size
            )
        else:
            assert self._dataset_type == "test"
            self.synsin_realestate10k = RealEstate10KEval(
                config.eval_test_frames, config.eval_test_cameras, config.image_size
            )

        # create a dummy vis mask for image central regions
        self.dummy_vis_mask = torch.zeros(
            config.image_size, config.image_size, 1, dtype=torch.float32
        )
        crop = config.image_size // 4
        self.dummy_vis_mask[crop:-crop, crop:-crop] = 1.

    def __getitem__(self, idx):
        synsin_data = self.synsin_realestate10k[idx]

        current_sample = Sample()
        current_sample.image_id = object_to_byte_tensor(synsin_data["video_id"])
        for n_view in range(self.num_view_per_sample):
            rgb = synsin_data["images"][n_view]
            camera_P = synsin_data["cameras"][n_view]["P"]
            camera_R, camera_T = get_pytorch3d_camera_RT(camera_P)

            orig_img = torch.tensor(rgb, dtype=torch.float32)
            trans_img = self.image_processor(orig_img.permute((2, 0, 1)))
            all_one_mask = torch.ones(orig_img.size(0), orig_img.size(1))
            setattr(current_sample, 'orig_img_{}'.format(n_view), orig_img)
            setattr(current_sample, 'trans_img_{}'.format(n_view), trans_img)
            setattr(current_sample, 'depth_mask_{}'.format(n_view), all_one_mask)

            # camera poses R and T
            R = torch.tensor(camera_R, dtype=torch.float32)
            T = torch.tensor(camera_T, dtype=torch.float32)
            setattr(current_sample, 'R_{}'.format(n_view), R)
            setattr(current_sample, 'T_{}'.format(n_view), T)

        current_sample.vis_mask = self.dummy_vis_mask

        return current_sample

    def __len__(self):
        return len(self.synsin_realestate10k)


def get_pytorch3d_camera_RT(P):
    P = P.copy()

    # change from Habitat coordinates to PyTorch3D coordinates
    P[0] *= -1  # flip X axis
    P[2] *= -1  # flip Z axis

    R = P[0:3, 0:3].T  # to row major
    T = P[0:3, 3]

    return R, T

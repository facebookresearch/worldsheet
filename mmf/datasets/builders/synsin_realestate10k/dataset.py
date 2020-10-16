# Copyright (c) Facebook, Inc. and its affiliates.
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
        if self._dataset_type == "train":
            opts = SynSinDatasetOption(
                config.train_data_dir, config.train_video_list, config.image_size
            )
            self.synsin_realestate10k = RealEstate10K(
                self._dataset_type, opts=opts, num_views=self.num_view_per_sample
            )
        else:
            self.synsin_realestate10k = RealEstate10KEval(
                config.eval_data_dir, config.eval_video_list, config.image_size
            )

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

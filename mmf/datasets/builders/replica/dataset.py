# Copyright (c) Facebook, Inc. and its affiliates.
import os
import skimage.io
import numpy as np
import torch
from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset


class ReplicaDataset(MMFDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__("replica", config, dataset_type, imdb_file_index, *args, **kwargs)

        self.multiview_data_dir = config.multiview_data_dir
        self.multiview_image_dir = config.multiview_image_dir
        self.num_view_per_sample = config.num_view_per_sample

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()
        current_sample.image_id = sample_info["image_id"]

        data_path = os.path.join(self.multiview_data_dir, sample_info["data_path"])
        data_f = np.load(data_path)
        for n_view in range(self.num_view_per_sample):
            # RGB image
            image_path = os.path.join(
                self.multiview_image_dir, sample_info["image_path_template"] % n_view
            )
            orig_img = torch.tensor(
                skimage.img_as_float(skimage.io.imread(image_path)), dtype=torch.float32
            )
            trans_img = self.image_processor(orig_img.permute((2, 0, 1)))
            setattr(current_sample, 'orig_img_{}'.format(n_view), orig_img)
            setattr(current_sample, 'trans_img_{}'.format(n_view), trans_img)

            # depth
            orig_depth = torch.tensor(data_f["depths"][n_view, ..., 0], dtype=torch.float32)
            depth_mask = orig_depth.gt(0)
            # fill invalid depth with a maximum depth (assuming they are far away)
            # invalid depth usually corresponds to holes or missing ceilings
            depth = orig_depth.clone().detach()
            depth.masked_fill_(~depth_mask, torch.max(orig_depth))
            depth = torch.clamp(depth, min=1e-2)
            setattr(current_sample, 'orig_depth_{}'.format(n_view), orig_depth)
            setattr(current_sample, 'depth_{}'.format(n_view), depth)
            setattr(current_sample, 'depth_mask_{}'.format(n_view), depth_mask)

            # camera poses R and T
            R = torch.tensor(data_f["camera_Rs"][n_view], dtype=torch.float32)
            T = torch.tensor(data_f["camera_Ts"][n_view], dtype=torch.float32)
            setattr(current_sample, 'R_{}'.format(n_view), R)
            setattr(current_sample, 'T_{}'.format(n_view), T)

        data_f.close()

        return current_sample

    def format_for_prediction(self, report):
        raise NotImplementedError()

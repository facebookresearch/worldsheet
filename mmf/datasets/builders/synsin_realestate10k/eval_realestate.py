# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. All rights reserved.

import numpy as np
import torch.utils.data as data
# from skimage.io import imread
# from skimage.transform import resize
from skimage import img_as_float
from PIL import Image


class Dataset(data.Dataset):
    def __init__(self, base_path, video_list_file, W):

        self.base_path = base_path
        self.files = np.loadtxt(video_list_file, dtype=np.str)

        self.offset = np.array(
            [[2, 0, -1], [0, -2, 1], [0, 0, -1]], dtype=np.float32
        )

        # self.K = np.array(
        #     [
        #         [1.0, 0.0, 0.0, 0.0],
        #         [0, 1.0, 0.0, 0.0],
        #         [0.0, 0.0, 1.0, 0.0],
        #         [0.0, 0.0, 0.0, 1.0],
        #     ],
        #     dtype=np.float32,
        # )
        # self.invK = np.linalg.inv(self.K)

        # self.input_transform = torchvision.transforms.Compose(
        #     [
        #         torchvision.transforms.Resize((W, W)),
        #         torchvision.transforms.ToTensor(),
        #         torchvision.transforms.Normalize(
        #             (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        #         ),
        #     ]
        # )

        self.W = W

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        # Then load the image and generate that
        file_name = self.files[index]

        src_image_name = (
            self.base_path
            + '/%s/%s.png' % (file_name[0], file_name[1])
        )
        tgt_image_name = (
            self.base_path
            + '/%s/%s.png' % (file_name[0], file_name[2])
        )

        intrinsics = file_name[3:7].astype(np.float32) / float(self.W)
        src_pose = file_name[7:19].astype(np.float32).reshape(3, 4)
        tgt_pose = file_name[19:].astype(np.float32).reshape(3, 4)

        # use PIL.Image.resize to be consistent w/ torchvision transforms
        # src_image = resize(imread(src_image_name), [self.W, self.W])
        # tgt_image = resize(imread(tgt_image_name), [self.W, self.W])
        src_image = img_as_float(
            np.array(
                Image.open(src_image_name).resize([self.W, self.W], Image.BILINEAR)))
        tgt_image = img_as_float(
            np.array(
                Image.open(tgt_image_name).resize([self.W, self.W], Image.BILINEAR)))

        poses = [src_pose, tgt_pose]
        cameras = []

        for pose in poses:

            origK = np.array(
                [
                    [intrinsics[0], 0, intrinsics[2]],
                    [0, intrinsics[1], intrinsics[3]],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            K = np.matmul(self.offset, origK)

            P = pose
            P = np.matmul(K, P)
            # Merge these together to match habitat
            P = np.vstack((P, np.zeros((1, 4)))).astype(np.float32)
            P[3, 3] = 1

            # Now artificially flip x/ys to match habitat
            # Pinv = np.linalg.inv(P)

            cameras += [{"P": P}]

            video_id = f"{index:04d}_{file_name[0]}_{file_name[1]}_{file_name[2]}"

        return {"images": [src_image, tgt_image], "cameras": cameras, "video_id": video_id}

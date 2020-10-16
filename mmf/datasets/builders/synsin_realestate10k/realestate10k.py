# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import numpy as np
import torch.utils.data as data
# from PIL import Image
from skimage.io import imread
from skimage.transform import resize
# from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from .geometry import get_deltas


class RealEstate10K(data.Dataset):
    """ Dataset for loading the RealEstate10K. In this case, images are randomly
    chosen within a video subject to certain constraints: e.g. they should
    be within a number of frames but the angle and translation should
    vary as much as possible.
    """

    def __init__(
        self, dataset, opts=None, num_views=2, seed=0, vectorize=False
    ):
        # Now go through the dataset

        self.imageset = np.loadtxt(
            opts.video_list,
            dtype=np.str,
        )

        if dataset == "train":
            self.imageset = self.imageset[0 : int(0.8 * self.imageset.shape[0])]
        else:
            self.imageset = self.imageset[int(0.8 * self.imageset.shape[0]) :]

        self.rng = np.random.RandomState(seed)
        self.base_file = opts.train_data_path

        self.num_views = num_views

        # self.input_transform = Compose(
        #     [
        #         Resize((opts.W, opts.W)),
        #         ToTensor(),
        #         Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #     ]
        # )
        self.W = opts.W

        self.offset = np.array(
            [[2, 0, -1], [0, -2, 1], [0, 0, -1]],  # Flip ys to match habitat
            dtype=np.float32,
        )  # Make z negative to match habitat (which assumes a negative z)

        self.dataset = "train"

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

        self.ANGLE_THRESH = 5
        self.TRANS_THRESH = 0.15

    def __len__(self):
        return len(self.imageset)

    def __getitem__(self, index):
        # index = self.rng.randint(self.imageset.shape[0])
        # index = index % self.imageset.shape[0]
        # Load text file containing frame information
        frames = np.loadtxt(
            self.base_file + "/%s.txt" % self.imageset[index]
        )

        image_index = self.rng.choice(frames.shape[0], size=(1,))[0]

        # Chose 15 images within 30 frames of the iniital one
        image_indices = self.rng.randint(80, size=(30,)) - 40 + image_index
        image_indices = np.minimum(
            np.maximum(image_indices, 0), frames.shape[0] - 1
        )

        # Look at the change in angle and choose a hard one
        angles = []
        translations = []
        for viewpoint in range(0, image_indices.shape[0]):
            orig_viewpoint = frames[image_index, 7:].reshape(3, 4)
            new_viewpoint = frames[image_indices[viewpoint], 7:].reshape(3, 4)
            dang, dtrans = get_deltas(orig_viewpoint, new_viewpoint)

            angles += [dang]
            translations += [dtrans]

        angles = np.array(angles)
        translations = np.array(translations)

        mask = image_indices[
            (angles > self.ANGLE_THRESH) | (translations > self.TRANS_THRESH)
        ]

        rgbs = []
        cameras = []
        for i in range(0, self.num_views):
            if i == 0:
                t_index = image_index
            elif mask.shape[0] > 5:
                # Choose a harder angle change
                t_index = mask[self.rng.randint(mask.shape[0])]
            else:
                t_index = image_indices[
                    self.rng.randint(image_indices.shape[0])
                ]

            image = imread(
                self.base_file
                + "/%s/" % (self.imageset[index])
                + str(int(frames[t_index, 0]))
                + ".png"
            )
            rgbs += [resize(image, [self.W, self.W])]

            intrinsics = frames[t_index, 1:7]
            extrinsics = frames[t_index, 7:]

            origK = np.array(
                [
                    [intrinsics[0], 0, intrinsics[2]],
                    [0, intrinsics[1], intrinsics[3]],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            K = np.matmul(self.offset, origK)

            origP = extrinsics.reshape(3, 4)
            P = np.matmul(K, origP)  # Merge these together to match habitat
            P = np.vstack((P, np.zeros((1, 4), dtype=np.float32))).astype(
                np.float32
            )
            P[3, 3] = 1

            # Pinv = np.linalg.inv(P)

            cameras += [
                {
                    "P": P,
                    # "Pinv": Pinv,
                    # "OrigP": origP,
                    # "K": self.K,
                    # "Kinv": self.invK,
                }
            ]

        return {"images": rgbs, "cameras": cameras, "video_id": self.imageset[index]}

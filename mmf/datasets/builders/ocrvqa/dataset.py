# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from mmf.datasets.builders.textvqa.dataset import TextVQADataset


class OCRVQADataset(TextVQADataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(config, dataset_type, imdb_file_index, *args, **kwargs)
        self.dataset_name = "ocrvqa"

    def preprocess_sample_info(self, sample_info):
        # Do nothing in this case
        return sample_info

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from mmf.datasets.builders.textvqa.dataset import TextVQADataset


class STVQADataset(TextVQADataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(config, dataset_type, imdb_file_index, *args, **kwargs)
        self.dataset_name = "stvqa"

    def preprocess_sample_info(self, sample_info):
        feature_path = sample_info["feature_path"]
        append = "train"

        if self.dataset_type == "test":
            append = "test_task3"

        if not feature_path.startswith(append):
            feature_path = append + "/" + feature_path

        sample_info["feature_path"] = feature_path
        return sample_info

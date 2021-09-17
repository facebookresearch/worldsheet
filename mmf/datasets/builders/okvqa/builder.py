# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from mmf.common.registry import registry
from mmf.datasets.builders.okvqa.dataset import OKVQADataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("okvqa")
class OKVQABuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="okvqa", dataset_class=OKVQADataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/okvqa/defaults.yaml"

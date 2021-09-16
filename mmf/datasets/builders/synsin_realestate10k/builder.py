# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from mmf.common.registry import registry
from mmf.datasets.builders.synsin_realestate10k.dataset import \
    SynSinRealEstate10KDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("synsin_realestate10k")
class SynSinRealEstate10KBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="synsin_realestate10k",
        dataset_class=SynSinRealEstate10KDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class)
        self.dataset_class = SynSinRealEstate10KDataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/synsin_realestate10k/defaults.yaml"

    def load(self, config, dataset_type, *args, **kwargs):
        return self.dataset_class(config, dataset_type, 0)

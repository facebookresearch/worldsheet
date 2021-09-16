# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from mmf.common.registry import registry
from mmf.datasets.builders.synsin_habitat.dataset import SynSinHabitatDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("synsin_habitat")
class SynSinHabitatBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="synsin_habitat", dataset_class=SynSinHabitatDataset, *args,
        **kwargs
    ):
        super().__init__(dataset_name, dataset_class)
        self.dataset_class = SynSinHabitatDataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/synsin_habitat/defaults.yaml"

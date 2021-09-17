# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from mmf.common.registry import registry
from mmf.datasets.builders.replica.dataset import ReplicaDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("replica")
class ReplicaBuilder(MMFDatasetBuilder):
    def __init__(self, dataset_name="replica", dataset_class=ReplicaDataset, *args, **kwargs):
        super().__init__(dataset_name, dataset_class)
        self.dataset_class = ReplicaDataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/replica/defaults.yaml"

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from mmf.common.registry import registry
from mmf.datasets.builders.gqa.builder import GQABuilder
from mmf.datasets.builders.gqa.masked_dataset import MaskedGQADataset


@registry.register_builder("masked_gqa")
class MaskedGQABuilder(GQABuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "masked_gqa"
        self.dataset_class = MaskedGQADataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/gqa/masked.yaml"

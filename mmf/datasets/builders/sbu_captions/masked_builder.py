# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from mmf.common.registry import registry
from mmf.datasets.builders.coco import MaskedCOCOBuilder

from .masked_dataset import MaskedSBUDataset


@registry.register_builder("masked_sbu")
class MaskedSBUBuilder(MaskedCOCOBuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "masked_sbu"
        self.set_dataset_class(MaskedSBUDataset)

    @classmethod
    def config_path(cls):
        return "configs/datasets/sbu_captions/masked.yaml"

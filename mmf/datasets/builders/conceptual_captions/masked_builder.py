# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from mmf.common.registry import registry
from mmf.datasets.builders.coco import MaskedCOCOBuilder

from .masked_dataset import MaskedConceptualCaptionsDataset


@registry.register_builder("masked_conceptual_captions")
class MaskedConceptualCaptionsBuilder(MaskedCOCOBuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "masked_conceptual_captions"
        self.set_dataset_class(MaskedConceptualCaptionsDataset)

    @classmethod
    def config_path(cls):
        return "configs/datasets/conceptual_captions/masked.yaml"

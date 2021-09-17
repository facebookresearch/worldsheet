# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from mmf.common.registry import Registry
from mmf.datasets.builders.ocrvqa.dataset import OCRVQADataset
from mmf.datasets.builders.textvqa.builder import TextVQABuilder


@Registry.register_builder("ocrvqa")
class OCRVQABuilder(TextVQABuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "ocrvqa"
        self.set_dataset_class(OCRVQADataset)

    @classmethod
    def config_path(cls):
        return "configs/datasets/ocrvqa/defaults.yaml"

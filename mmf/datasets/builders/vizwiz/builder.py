# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from mmf.common.registry import registry
from mmf.datasets.builders.vizwiz.dataset import VizWizDataset
from mmf.datasets.builders.vqa2 import VQA2Builder


@registry.register_builder("vizwiz")
class VizWizBuilder(VQA2Builder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "vizwiz"
        self.set_dataset_class(VizWizDataset)

    @classmethod
    def config_path(cls):
        return "configs/datasets/vizwiz/defaults.yaml"

    def update_registry_for_model(self, config):
        super().update_registry_for_model(config)

# Copyright (c) Facebook, Inc. and its affiliates.

from mmf.common.registry import registry
from mmf.datasets.builders.diode.dataset import DiodeDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("diode")
class DiodeBuilder(MMFDatasetBuilder):
    def __init__(self, dataset_name="diode", dataset_class=DiodeDataset, *args, **kwargs):
        super().__init__(dataset_name, dataset_class)
        self.dataset_class = DiodeDataset

    @classmethod
    def config_path(cls):
        return "configs/datasets/diode/defaults.yaml"

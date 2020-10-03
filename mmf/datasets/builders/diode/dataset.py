# Copyright (c) Facebook, Inc. and its affiliates.
from mmf.datasets.builders.replica.dataset import ReplicaDataset


class DiodeDataset(ReplicaDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(config, dataset_type, imdb_file_index, *args, **kwargs)
        self._dataset_name = 'diode'

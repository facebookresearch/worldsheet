# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from mmf.datasets.databases.annotation_database import AnnotationDatabase


class SceneGraphDatabase(AnnotationDatabase):
    def __init__(self, config, scene_graph_path, *args, **kwargs):
        super().__init__(config, scene_graph_path, *args, **kwargs)
        self.data_dict = {}
        for item in self.data:
            self.data_dict[item["image_id"]] = item

    def __getitem__(self, idx):
        return self.data_dict[idx]

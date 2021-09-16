# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from mmf.common.batch_collator import BatchCollator
from mmf.common.sample import Sample

import tests.test_utils as test_utils


class TestBatchCollator(unittest.TestCase):
    def test_call(self):
        batch_collator = BatchCollator("vqa2", "train")
        sample_list = test_utils.build_random_sample_list()
        sample_list = batch_collator(sample_list)

        # Test already build sample list
        self.assertEqual(sample_list.dataset_name, "vqa2")
        self.assertEqual(sample_list.dataset_type, "train")

        sample = Sample()
        sample.a = torch.tensor([1, 2], dtype=torch.int)

        # Test list of samples
        sample_list = batch_collator([sample, sample])
        self.assertTrue(
            test_utils.compare_tensors(
                sample_list.a, torch.tensor([[1, 2], [1, 2]], dtype=torch.int)
            )
        )

        # Test IterableDataset case
        sample_list = test_utils.build_random_sample_list()
        new_sample_list = batch_collator([sample_list])
        self.assertEqual(new_sample_list, sample_list)

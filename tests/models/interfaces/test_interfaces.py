# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
from mmf.models.mmbt import MMBT

import tests.test_utils as test_utils


class TestModelInterfaces(unittest.TestCase):
    @test_utils.skip_if_no_network
    @test_utils.skip_if_windows
    @test_utils.skip_if_macos
    def test_mmbt_hm_interface(self):
        model = MMBT.from_pretrained("mmbt.hateful_memes.images")
        result = model.classify(
            "https://i.imgur.com/tEcsk5q.jpg", "look how many people love you"
        )

        self.assertEqual(result["label"], 0)
        np.testing.assert_almost_equal(result["confidence"], 0.9993, decimal=4)
        result = model.classify(
            "https://i.imgur.com/tEcsk5q.jpg", "they have the privilege"
        )
        self.assertEqual(result["label"], 0)
        np.testing.assert_almost_equal(result["confidence"], 0.9777, decimal=4)
        result = model.classify("https://i.imgur.com/tEcsk5q.jpg", "hitler and jews")
        self.assertEqual(result["label"], 1)
        np.testing.assert_almost_equal(result["confidence"], 0.6342, decimal=4)

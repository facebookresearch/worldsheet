# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import time
import unittest

from mmf.utils.timer import Timer


class TestUtilsTimer(unittest.TestCase):
    def test_get_current(self):
        timer = Timer()
        expected = 0

        self.assertEqual(int(timer.get_current().split("ms")[0]), expected)

    def test_reset(self):
        timer = Timer()
        time.sleep(2)
        timer.reset()
        expected = 0

        self.assertEqual(int(timer.get_current().split("ms")[0]), expected)

    def test_get_time_since_start(self):
        timer = Timer()
        time.sleep(2)
        expected = 2

        self.assertEqual(expected, int(timer.get_time_since_start().split("s")[0]))

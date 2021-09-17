# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from abc import ABC
from typing import Type

from mmf.utils.timer import Timer


logger = logging.getLogger(__name__)


class TrainerProfilingMixin(ABC):
    profiler: Type[Timer] = Timer()

    def profile(self, text: str) -> None:
        if self.training_config.logger_level != "debug":
            return
        logging.debug(f"{text}: {self.profiler.get_time_since_start()}")
        self.profiler.reset()

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from mmf.trainers.callbacks.base import Callback
from mmf.utils.build import build_scheduler


class LRSchedulerCallback(Callback):
    """Callback which executes a LR scheduler. It is executed after every
    batch iteration.
    """

    def __init__(self, config, trainer):
        """
        Attr:
            config(mmf_typings.DictConfig): Config for the callback
            trainer(Type[BaseTrainer]): Trainer object
        """
        super().__init__(config, trainer)

        self._scheduler = None
        if self.training_config.lr_scheduler is True:
            self._scheduler = build_scheduler(self.trainer.optimizer, self.config)

    def on_update_end(self, **kwargs):
        if self._scheduler is not None:
            self._scheduler.step()

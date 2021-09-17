# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from mmf.common.registry import registry
from transformers.optimization import AdamW


registry.register_optimizer("adam_w")(AdamW)

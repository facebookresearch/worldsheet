# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


def transform_to_batch_sequence(tensor):
    if tensor is not None:
        if len(tensor.size()) == 2:
            return tensor
        else:
            assert len(tensor.size()) == 3
            return tensor.contiguous().view(-1, tensor.size(-1))
    else:
        return None


def transform_to_batch_sequence_dim(tensor):
    if tensor is not None:
        if len(tensor.size()) == 3:
            return tensor
        else:
            assert len(tensor.size()) == 4
            return tensor.contiguous().view(-1, tensor.size(-2), tensor.size(-1))
    else:
        return None

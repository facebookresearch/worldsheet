# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys


__version__ = "1.0.0rc11"

msg = "MMF is only compatible with Python 3.6 and newer."


if sys.version_info < (3, 6):
    raise ImportError(msg)

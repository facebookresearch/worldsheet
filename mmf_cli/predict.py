#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

from mmf_cli.run import run


def predict(opts=None):
    if opts is None:
        sys.argv.extend(["evaluation.predict=true"])
    else:
        opts.extend(["evaluation.predict=true"])

    run(predict=True)


if __name__ == "__main__":
    predict()

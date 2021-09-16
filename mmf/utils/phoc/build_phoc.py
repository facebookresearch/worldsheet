# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np

from .cphoc import build_phoc as _build_phoc_raw


_alphabet = {
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
}  # NoQA


def build_phoc(token):
    token = token.lower().strip()
    token = "".join([c for c in token if c in _alphabet])
    phoc = _build_phoc_raw(token)
    phoc = np.array(phoc, dtype=np.float32)
    return phoc

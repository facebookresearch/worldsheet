# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import json

from tools.scripts.gqa.extract_vocabulary import ExtractVocabulary


class ExtractVisdialVocabulary(ExtractVocabulary):
    def __init__(self):
        super().__init__()

    def get_text(self):
        text = []

        for input_file in self.input_files:
            with open(input_file) as f:
                f_json = json.load(f)
                # Add 'questions' from visdial
                text += f_json["data"]["questions"]
                # Add 'answers' from visdial
                text += f_json["data"]["answers"]

                for dialog in f_json["data"]["dialogs"]:
                    text += [dialog["caption"]]
        return text


if __name__ == "__main__":
    extractor = ExtractVisdialVocabulary()
    extractor.extract()

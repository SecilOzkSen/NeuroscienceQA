# Copyright 2021 Open Business Software Solutions, the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test fixture for version 1.1. of SQUAD: The Stanford Question Answering Dataset.
This implementation is adapted from the question answering pipeline from the
HuggingFace's transformers library. Original code is available at:
`<https://https://github.com/huggingface/datasets/blob/master/datasets/squad/squad.py>`_.

"""

import json

import datasets
import os

logger = datasets.logging.get_logger(__name__)

_DESCRIPTION = """\
Neuroscience dataset.
"""

WORKING_DIR = os.getcwd()


class PolicyQATestFixtureConfig(datasets.BuilderConfig):
    """
    BuilderConfig for SQUAD test data.
    Args:
        **kwargs ():
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class PolicyQATextFixture(datasets.GeneratorBasedBuilder):
    """A test dataset taken from SQUAD Version 1.1.  for trapper's QA modules"""

    BUILDER_CONFIGS = [
        PolicyQATestFixtureConfig(
            name="policy_qa_test_fixture",
            version=datasets.Version("1.0.0", ""),
            description="PolicyQA test fixtures",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "label": datasets.Value("int32"),
                    "question": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                    "short_answer_in_sentence": datasets.Value("bool"),
                    "sentence_in_long_answer": datasets.Value("bool")
                }
            ),
            supervised_keys=None,
            homepage="",
        )

    def _download_fixture_data(self):
        with open("dataset/dev/dev.json", 'r') as dev_f:
            dev_data = json.load(dev_f)
        with open("dataset/train/train.json", 'r') as train_f:
            train_data = json.load(train_f)
        dataset = {}
        dataset["train"] = train_data
        dataset["dev"] = dev_data
        return dataset

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": f"{WORKING_DIR}/policyqa_text_fixture/dataset/train/train.json"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": f"{WORKING_DIR}/policyqa_text_fixture/dataset/dev/dev.json"},
            ),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        key = 0
        with open(filepath, encoding="utf-8") as f:
            json_str = json.load(f)
            for data in json_str:
                label = data.get("label", 0)
                question = data.get("question", "")
                sentence = data.get("sentence", "")
                short_answer_in_sentence = data.get("short_answer_in_sentence", )
                sentence_in_long_answer = data.get("sentence_in_long_answer", )
                yield key, {
                            "label": label,
                            "question": question,
                            "sentence": sentence,
                            "short_answer_in_sentence": short_answer_in_sentence,
                            "sentence_in_long_answer": sentence_in_long_answer,
                        }
                key += 1
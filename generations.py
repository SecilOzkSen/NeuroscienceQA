from typing import Union, Dict, List
from trapper.common.io import json_load, json_save
from copy import deepcopy
import os
from trapper.pipelines.question_answering_pipeline import SquadQuestionAnsweringPipeline
from trapper.pipelines.pipeline import create_pipeline_from_checkpoint
from transformers import pipeline

from jury import Jury

def prepare_samples(data: Union[str, Dict]):
    if isinstance(data, str):
        data = json_load(data)
    data = data["data"]
    qa_samples = []

    for article in data:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                sample = {}
                sample["context"] = paragraph["context"]
                sample["question"] = qa["question"]
                sample["gold_answers"] = [ans["text"] for ans in qa["answers"]]
                qa_samples.append(sample)

    return qa_samples

def prepare_samples_hf(data: Union[str, Dict]):
    if isinstance(data, str):
        data = json_load(data)
    data = data["data"]
    qa_samples = []

    for article in data:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                sample = {}
                sample["context"] = paragraph["context"]
                if len(qa["generated_questions"]) == 0:
                    continue
                sample["question"] = qa["generated_questions"][0]
                sample["gold_answers"] = []
                for ans in qa["qa_answers"]:
                    if len(ans["handle_impossible_answer_True"]["answer"]) == 0:
                        if len(ans["handle_impossible_answer_False"]["answer"]) == 0:
                            continue
                        else:
                            sample["gold_answers"].append(ans["handle_impossible_answer_False"]["answer"])
                    else:
                        sample["gold_answers"].append(ans["handle_impossible_answer_True"]["answer"])
                qa_samples.append(sample)

    return qa_samples


def prepare_samples_for_pipeline(samples: List[Dict]):
    pipeline_samples = deepcopy(samples)
    for i, sample in enumerate(pipeline_samples):
        sample.pop("gold_answers")
        if "id" not in sample:
            sample["id"] = str(i)
    return pipeline_samples

def prepare_samples_for_pipeline_hf(samples: List[Dict]):
    pipeline_samples = deepcopy(samples)
    for i, sample in enumerate(pipeline_samples):
        sample.pop("gold_answers")
    return pipeline_samples


def predict(pipeline, samples, **kwargs):
    pipeline_samples = prepare_samples_for_pipeline(samples)
    predictions = pipeline(pipeline_samples, **kwargs)
    for i, prediction in enumerate(predictions):
        if prediction is List:
            prediction = prediction[0]
        samples[i]["predicted_answer"] = prediction["answer"].text
    return samples

def predict_hf(pipeline, samples):
    pipeline_samples = prepare_samples_for_pipeline_hf(samples)
    predictions = pipeline(pipeline_samples)
    for i, prediction in enumerate(predictions):
        if prediction is List:
            prediction = prediction[0]
        samples[i]["predicted_answer"] = prediction["answer"]
    return samples


WORKING_DIR = os.getcwd()
EXPORT_PATH = os.path.join(WORKING_DIR, "roberta-base-squad2-test-outputs.json")

TEST_SET = "/home/secilsen/PycharmProjects/NeuroscienceQA/neuroscience_test_fixture/dataset/test/test.json"
PRETRAINED_MODEL_PATH = "/home/secilsen/PycharmProjects/NeuroscienceQA/roberta-large-squad2/outputs"
EXPERIMENT_CONFIG = "/home/secilsen/PycharmProjects/NeuroscienceQA/roberta-large-squad2/outputs/experiment_config.json"




# a) Get predictions

if __name__ == '__main__':
    '''  ###### huggingface models
       model_name = "deepset/tinyroberta-squad2"
       nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
       samples = prepare_samples_hf(TEST_SET)
       predictions = predict_hf(nlp, samples) '''
    ####### trained models
    qa_pipeline = create_pipeline_from_checkpoint(
        checkpoint_path=PRETRAINED_MODEL_PATH,
        experiment_config_path=EXPERIMENT_CONFIG,
        task="squad-question-answering",
    )
    samples = prepare_samples_hf(TEST_SET)
    predictions = predict(qa_pipeline, samples)

    json_save(predictions, "/home/secilsen/PycharmProjects/NeuroscienceQA/roberta-large-squad2/roberta-tiny-squad2-trained-predictions.json")
    references = [sample["gold_answers"] for sample in predictions]
    hypotheses = [sample["predicted_answer"] for sample in predictions]
    jury = Jury(metrics="squad")
    evaluations = jury.evaluate(references=references, predictions=hypotheses)
    json_save(evaluations,
              "/home/secilsen/PycharmProjects/NeuroscienceQA/roberta-large-squad2/roberta-tiny-squad2-trained-evaluations.json")
    print(evaluations)
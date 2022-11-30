import json
import multiprocessing
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import transformers
import os

from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import Trainer, TrainingArguments
from transformers import DefaultDataCollator
from transformers.trainer_callback import EarlyStoppingCallback
from transformers import AdamW

# HYPERPARAMETERS

SEED_SPLIT = 0
SEED_TRAIN = 0

MAX_SEQ_LEN = 500
TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2
LEARNING_RATE = 2e-5
LR_WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
MODEL_PATH = "/home/secilsen/PycharmProjects/NeuroscienceQA/roberta-base-asnq"

with open("/home/secilsen/PycharmProjects/NeuroscienceQA/policyqa_text_fixture/dataset/train/train.json", 'r', encoding='utf-8') as f:
    train_json = json.load(f)

with open("/home/secilsen/PycharmProjects/NeuroscienceQA/policyqa_text_fixture/dataset/dev/dev.json", 'r', encoding='utf-8') as f:
    valid_json = json.load(f)

train_dataset_df = pd.DataFrame.from_dict(train_json)
valid_dataset_df = pd.DataFrame.from_dict(valid_json)

train_dataset_df = train_dataset_df.drop(labels=['sentence_in_long_answer', 'short_answer_in_sentence'], axis=1)
valid_dataset_df = valid_dataset_df.drop(labels=['sentence_in_long_answer', 'short_answer_in_sentence'], axis=1)

train_dataset = Dataset.from_pandas(train_dataset_df)
valid_dataset = Dataset.from_pandas(valid_dataset_df)

tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_PATH, max_length=MAX_SEQ_LEN)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)

def tokenize(row):
    list_with_sep = []
    for q, s in zip(row['question'], row['sentence']):
        list_with_sep.append(f"<s> {q} </s> {s} </s>")
    return tokenizer(list_with_sep,
                     padding=True,
                     truncation=True,
                     add_special_tokens=False)

train_dataset = train_dataset.map(tokenize, batched=True, batch_size=len(train_dataset))

valid_dataset = valid_dataset.map(tokenize, batched=True, batch_size=len(valid_dataset))

train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
valid_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

os.environ["WANDB_DISABLED"] = "true"

training_args = TrainingArguments(
    output_dir = '/home/secilsen/PycharmProjects/NeuroscienceQA/roberta-large-asnq-policyqa',
    num_train_epochs=50,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=12,
    per_device_eval_batch_size=2,
    evaluation_strategy = "epoch",
    save_strategy="epoch",
    gradient_checkpointing=True,
    disable_tqdm=False,
    load_best_model_at_end=True,
    warmup_steps=500,
    logging_steps=500,
    fp16=True,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    optim='adamw_hf',
    metric_for_best_model='f1',
    logging_dir='/home/secilsen/PycharmProjects/NeuroscienceQA/policyqa_text_fixture/logging',
    run_name = 'roberta-large-asnq-policyqa-classification',
)

if __name__ == '__main__':
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,

    )
    result = trainer.train()


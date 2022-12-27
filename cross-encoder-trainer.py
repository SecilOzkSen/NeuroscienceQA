import json
import pandas as pd
import os
from datasets import Dataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator

from transformers import TrainingArguments
import logging
from sentence_transformers import LoggingHandler, util
from sentence_transformers import InputExample, CrossEncoder
import math

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout

# HYPERPARAMETERS

SEED_SPLIT = 0
SEED_TRAIN = 0

MAX_SEQ_LEN = 500
TRAIN_BATCH_SIZE = 5
LEARNING_RATE = 2e-5
LR_WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 50
BATCH_SIZE = 8

def create_cross_encoder_dataset(contexes_list, questions_list, negatives_per_example = 7):
    data_size = len(questions_list)
    data_list = []
    for index, (question, context) in enumerate(zip(questions_list, contexes_list)):
        batch = []
        for i in range(negatives_per_example):
            negatives_index = index + 1
            context_list_index = 0 if negatives_index >= data_size - 1 else negatives_index
            context = contexes_list[context_list_index]
            batch.append(InputExample(texts=[question, context], label=0))
        data_list.extend(batch)
    return data_list

os.environ["WANDB_DISABLED"] = "true"


if __name__ == '__main__':
    full_df = pd.read_csv(
        'policyQA_bsbs_sentence.csv', delimiter='|')
    full_df = full_df.drop(labels=['answer'], axis=1)
    train_context, valid_context, train_question, valid_question = train_test_split(full_df['context'].tolist(),
                                                                                    full_df['question'].tolist(),
                                                                                    test_size=0.01, random_state=8)
    train_dataset = create_cross_encoder_dataset(train_context, train_question) #in-batch negatives implemented.
    valid_dataset = create_cross_encoder_dataset(valid_context, valid_question)  # in-batch negatives implemented.
    model = CrossEncoder('roberta-base', num_labels=1)
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=BATCH_SIZE)
    evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(valid_dataset, name='policyqa-dev')

    warmup_steps = math.ceil(len(train_dataloader) * NUM_EPOCHS * 0.1)  # 10% of train data for warm-up
    logger.info("Warmup-steps: {}".format(warmup_steps))
    model.fit(train_dataloader=train_dataloader,
              evaluator=evaluator,
              epochs=NUM_EPOCHS,
              evaluation_steps=len(train_dataset),
              warmup_steps=warmup_steps,
              output_path='/home/secilsen/PycharmProjects/NeuroscienceQA/cross-encoder-model')





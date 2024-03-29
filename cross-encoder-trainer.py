
import pandas as pd
import os
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
import logging
from sentence_transformers import LoggingHandler, util
from sentence_transformers import InputExample, CrossEncoder, losses
import math
from sentence_transformers import SentenceTransformer
import pickle
from difflib import SequenceMatcher
import torch
from torch import nn
import numpy as np
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
BATCH_SIZE = 6

bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
bi_encoder.max_seq_length = 500
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
with open('data/st-context-embeddings.pkl', "rb") as fIn:
    cache_data = pickle.load(fIn)
    corpus_sentences = cache_data['contexes']
    corpus_embeddings = cache_data['embeddings']
cross_entropy_loss = nn.CrossEntropyLoss()

def search(question):
    # Semantic Search (Retrieve)
    question_embedding = bi_encoder.encode(question, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=100)
    if len(hits) == 0:
        return []
    hits = hits[0]
    # Rerank - score all retrieved passages with cross-encoder
    cross_inp = [[question, corpus_sentences[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    # Output of top-5 hits from re-ranker
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    top_5_contexes = []
    top_5_scores = []
    for hit in hits[:20]:
        top_5_contexes.append(corpus_sentences[hit['corpus_id']])
        top_5_scores.append(hit['cross-score'])
    return top_5_contexes, top_5_scores

def create_cross_encoder_dataset_hard_negatives(contexes_list, questions_list, negatives_per_example = 5, upper_bound=5):
    data_list = []
    for index, (question, context) in enumerate(zip(questions_list, contexes_list)):
        batch = [InputExample(texts=[question, context], label=1)]
        top_20_contexes, _ = search(question)
        for negative_context in top_20_contexes[upper_bound:upper_bound+negatives_per_example]:
            if context.strip() == negative_context.strip():
                continue
            batch.append(InputExample(texts=[question, negative_context], label=0))
        data_list.extend(batch)
    return data_list

def create_cross_encoder_dataset(contexes_list, questions_list, indexes_list, negatives_per_example = 5):
    data_size = len(questions_list)
    data_list = []
    for index, (question, context, context_index) in enumerate(zip(questions_list, contexes_list, indexes_list)):
        batch = []
        batch.append(InputExample(texts=[question, context], label=1))
        negatives_index = index
        for i in range(negatives_per_example):
            negative_context_index = context_index
            while context_index == negative_context_index:
                negatives_index += 1
                context_list_index = 0 if negatives_index >= data_size - 1 else negatives_index
                negative_context_index = indexes_list[context_list_index]
            negative_context = contexes_list[context_list_index]
            batch.append(InputExample(texts=[question, negative_context], label=0))
        data_list.extend(batch)
    return data_list

def contrastive_tension_loss_in_batch_negatives(scores, labels):
    softmax_op = nn.LogSoftmax(dim=0)
    scores = scores / 0.1
    positive_index = np.where(labels.detach().cpu().numpy() == 1)
    softmax_out = softmax_op(scores)
    loss = -softmax_out[positive_index]
    return loss[0]


os.environ["WANDB_DISABLED"] = "true"

if __name__ == '__main__':
    full_df = pd.read_csv(
        'data/basecamp.csv', delimiter='|')
    full_df = full_df.drop(labels=['answer'], axis=1)
    indexes = list(range(len(full_df.index)))
    train_index, valid_index, train_question, valid_question = train_test_split(indexes,
                                                                                    full_df['question'].tolist(),
                                                                                    test_size=0.1, random_state=8)
    context_list = full_df['context'].tolist()
    train_context = [context_list[i] for i in train_index]
    valid_context = [context_list[i] for i in valid_index]
    train_dataset = create_cross_encoder_dataset_hard_negatives(train_context, train_question, negatives_per_example=BATCH_SIZE-1) #in-batch negatives implemented.
    valid_dataset = create_cross_encoder_dataset_hard_negatives(valid_context, valid_question, negatives_per_example=BATCH_SIZE-1)  # in-batch negatives implemented.
    model = CrossEncoder('facebook/contriever-msmarco', num_labels=1)
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=BATCH_SIZE)
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(valid_dataset, name='basecamp-valid')

    warmup_steps = math.ceil(len(train_dataloader) * NUM_EPOCHS * 0.1)  # 10% of train data for warm-up
    logger.info("Warmup-steps: {}".format(warmup_steps))
    model.fit(train_dataloader=train_dataloader,
              loss_fct=contrastive_tension_loss_in_batch_negatives,
              evaluator=evaluator,
              epochs=NUM_EPOCHS,
              evaluation_steps=len(valid_dataset),
              warmup_steps=warmup_steps,
              output_path='cross-encoder-model-contrastive')





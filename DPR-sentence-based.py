import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from transformers import Trainer, TrainingArguments, TrainerCallback
from torch.utils.data import Dataset
import numpy as np
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from typing import Union, List, Dict
import pickle
import spacy
from difflib import SequenceMatcher
import pandas as pd
import random
import os
from sklearn.model_selection import train_test_split
from transformers import DPRContextEncoderTokenizer, \
    DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRQuestionEncoder, PreTrainedModel, PretrainedConfig

os.environ["WANDB_DISABLED"] = "true"

MODEL_DIR = 'DPR-model-contrastive-sentence'
BATCH_SIZE = 16

bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
bi_encoder.max_seq_length = 500
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
with open('data/st-context-embeddings.pkl', "rb") as fIn:
    cache_data = pickle.load(fIn)
    corpus_sentences = cache_data['contexes']
    corpus_embeddings = cache_data['embeddings']

nlp = spacy.load("en_core_web_sm")

context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')


class CustomDPRConfig(PretrainedConfig):
    model_type = 'dpr'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class HFDPRModel(PreTrainedModel):
    config_class = CustomDPRConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = DPRModel()

    def forward(self, input):
        return self.model(input)


def search(question, negatives_num, upper_bound=5):
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
    for hit in hits[upper_bound:]:
        top_5_contexes.append(corpus_sentences[hit['corpus_id']])
        top_5_scores.append(hit['cross-score'])
    return top_5_contexes, top_5_scores


class DPRModel(nn.Module):
    def __init__(self,
                 question_model_name='facebook/dpr-question_encoder-single-nq-base',
                 context_model_name='facebook/dpr-ctx_encoder-single-nq-base'):
        super(DPRModel, self).__init__()
        self.question_model = DPRQuestionEncoder.from_pretrained(question_model_name)
        self.context_model = DPRContextEncoder.from_pretrained(context_model_name)

    def batch_dot_product(self, context_output, question_output):
        mat1 = torch.unsqueeze(question_output, dim=1)
        mat2 = torch.unsqueeze(context_output, dim=2)
        result = torch.bmm(mat1, mat2)
        result = torch.squeeze(result, dim=1)
        result = torch.squeeze(result, dim=1)
        return result

    def forward(self, batch: Union[List[Dict], Dict]):
        context_tensor = batch['context_tensor']
        question_tensor = batch['question_tensor']
        context_model_output = self.context_model(input_ids=context_tensor['input_ids'],
                                                  attention_mask=context_tensor['attention_mask'])  # (bsz, hdim)
        question_model_output = self.question_model(input_ids=question_tensor['input_ids'],
                                                    attention_mask=question_tensor['attention_mask'])
        embeddings_context = context_model_output['pooler_output']
        embeddings_question = question_model_output['pooler_output']

        scores = self.batch_dot_product(embeddings_context, embeddings_question)  # self.scale
        return scores


def tokenize(batch: Union[List[Dict], Dict]):
    contextes = []
    questions = []
    labels = []
    if isinstance(batch, List):
        for item in batch:
            contextes.append(item['sentence'])
            questions.append(item['question'])
            labels.append(torch.tensor(item['label'], dtype=torch.int64))
    else:
        contextes.append(batch['sentence'])
        questions.append(batch['question'])
        labels.append(torch.Tensor(batch['labels']))

    context_tensor = context_tokenizer(contextes, padding=True, truncation=True, return_tensors="pt",
                                       add_special_tokens=True)
    question_tensor = question_tokenizer(questions, padding=True, truncation=True, return_tensors="pt",
                                         add_special_tokens=True)
    return [context_tensor, question_tensor], torch.stack(labels)

class DPRDataCollator:
    def collate_function(self, data):
        tensors, labels = data[0]
        context_tensor, question_tensor = tensors
        return dict(context_tensor=context_tensor,
                    #      context_attention_mask=context_tensor['attention_mask'],
                    question_tensor=question_tensor,
                    #       question_attention_mask=question_tensor['attention_mask'],
                    labels=labels)

    def __call__(self, data):
        return self.collate_function(data)


class CustomDPRDataset(Dataset):
    def __init__(self,
                 contexes_list,
                 questions_list,
                 answer_start_list,
                 batch_size=BATCH_SIZE,
                 negative_answer_in_positives=1,
                 upper_bound=5
                 ):
        self.contexes_list = contexes_list
        self.questions_list = questions_list
        self.answer_start_list = answer_start_list
        if len(self.contexes_list) != len(self.questions_list) != len(answer_start_list):
            raise ValueError("Length of context list and question list should be equal!")

        self.data_size = len(self.contexes_list)
        self.batch_size = batch_size
        self.negative_answer_in_positives = negative_answer_in_positives
        self.upper_bound = upper_bound

    def generate_negative_examples(self, sentence_list, question, context, negative_answer_size):
        sentences_length = len([sentence.text for sentence in sentence_list])
        if sentences_length == 1:
            return [dict(label=0,
                         question=question,
                         sentence=sentence_list[0].text,
                         context=context)]
        if sentences_length - 1 < negative_answer_size:
            negative_answer_size = sentences_length - 1
        indexes_list = [i for i in range(sentences_length)]
        rng = np.random.default_rng()
        previous_index = -1
        indexes = []
        while len(indexes) != negative_answer_size:
            try:
                index = rng.choice(indexes_list, 1)[0]
            except ValueError:
                print(indexes_list)
            if index == previous_index:
                continue
            indexes.append(index)
            previous_index = index
        data_list = []
        for index in indexes:
            data = dict(label=0,
                        question=question,
                        sentence=sentence_list[index].text,
                        context=context)
            data_list.append(data)

        return data_list

    def __getitem__(self, index):
        question = self.questions_list[index]
        answer_start = self.answer_start_list[index]
        context = self.contexes_list[index]
        sentences = nlp(context)
        batch = []
        sentence_list = list(sentences.sents)
        for sentence in sentence_list:
            if sentence.start_char <= answer_start < sentence.end_char:
                break
        data = dict(label=1,
                    question=question,
                    sentence=sentence.text,
                    context=context,
                    )
        batch.append(data)
        sentence_list.remove(sentence)
        negatives = []
        if len(sentence_list) != 0:
            negative_answer_in_positives = self.negative_answer_in_positives if self.negative_answer_in_positives <= len(sentence_list) else len(sentence_list)
            negatives = self.generate_negative_examples(sentence_list, question, context, negative_answer_in_positives)
        batch.extend(negatives)
        hard_negatives = BATCH_SIZE - len(batch) - 1
        top_contexes, _ = search(question, hard_negatives, self.upper_bound)
        for negative_context in top_contexes:
            if SequenceMatcher(context, negative_context).ratio() >= 0.9:
                continue
            negative_sentences = nlp(negative_context)
            negatives = self.generate_negative_examples(list(negative_sentences.sents), question, negative_context,
                                                        hard_negatives)
            if len(negatives) != 0:
                hard_negatives -= len(negatives)
            batch.extend(negatives)
            if len(batch) >= BATCH_SIZE:
                break
        tensors, labels = tokenize(batch[:BATCH_SIZE])
        return tensors, labels

    def __len__(self):
        return len(self.contexes_list)


class DPRTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def contrastive_tension_loss_in_batch_negatives(self, scores, labels):
        positive_index = np.where(labels.detach().cpu().numpy() == 1)
        res = torch.logsumexp(scores, dim=0) - (scores[positive_index[0]] * torch.log(torch.exp(torch.tensor(1))))
        return res[0]

    def compute_loss(self, model, inputs, return_outputs=False):
        scores = model(inputs)
        loss = self.contrastive_tension_loss_in_batch_negatives(scores, inputs['labels'])
        return (loss, scores) if return_outputs else loss


def dpr_valid_pipeline(input, dprModel):
    context = input['context']
    question = input['question']
    sentences = nlp(context)
    scores_list = []
    for sentence in sentences.sents:
        context_tensor = context_tokenizer(sentence.text, padding=True, truncation=True, return_tensors="pt",
                                           add_special_tokens=True)
        question_tensor = question_tokenizer(question, padding=True, truncation=True, return_tensors="pt",
                                             add_special_tokens=True)
        score = dprModel(dict(context_tensor=context_tensor.to('cuda:0'), question_tensor=question_tensor.to('cuda:0')))
        scores_list.append(score.detach().cpu().numpy()[0])
    return sum(scores_list) / len(scores_list)


def calculate_topk_accuracy(contexes, questions, dprModel, K_range, random_question_size=20):
    acc_dic = {}
    accuracies = np.zeros(shape=(len(questions), K_range))
    for index_q, question in enumerate(questions):
        scores_list = []
        random_indexes = random.sample(range(len(contexes)), random_question_size)
        random_indexes.append(index_q)
        random_indexes = list(set(random_indexes))
        gt_labels = np.zeros(shape=len(random_indexes))
        for i, random_index in enumerate(random_indexes):
            if random_index == index_q:
                gt_labels[i] = 1
            input_dict = dict(context=contexes[random_index], question=question)
            score = dpr_valid_pipeline(input_dict, dprModel)
            scores_list.append(score)
        sorted_indexes = sorted(range(len(scores_list)), key=lambda k: scores_list[k], reverse=True)
        # Top K ranked accuracy calculation
        for _k in range(K_range):
            acc = 0.
            for index in sorted_indexes[:_k + 1]:
                if gt_labels[index] == 1:
                    acc = 1.
                    break
            accuracies[index_q, _k] = acc
    for j in range(K_range):
        acc = np.mean(accuracies[:, j])
        acc_dic[f'top_{j + 1}_accuracy'] = acc
    return acc_dic


class DPRValidationCallback(TrainerCallback):
    def __init__(self,
                 valid_contexes,
                 valid_questions,
                 K=5,
                 rand_question_selected=20):
        self.validation_scores = []
        self.K = K
        self.rand_question_selected = rand_question_selected
        self.valid_contexes = valid_contexes
        self.valid_questions = valid_questions

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        metrics = kwargs.get("metrics")
        accuracy_dict = calculate_topk_accuracy(self.valid_contexes, self.valid_questions, model, K_range=self.K,
                                                random_question_size=self.rand_question_selected)
        metrics['eval_top_1_accuracy'] = accuracy_dict['top_1_accuracy']
        state.log_history[-1]['eval_accuracies'] = accuracy_dict


if __name__ == '__main__':
    full_df = pd.read_csv(
        'data/basecamp.csv', delimiter='|')
    #  full_df = full_df.drop(labels=['answer'], axis=1)
    indexes = list(range(len(full_df.index)))

    train_index, valid_index, train_question, valid_question = train_test_split(indexes,
                                                                                full_df['question'].tolist(),
                                                                                test_size=0.1, random_state=8)
    context_list = full_df['context'].tolist()
    answer_start_list = full_df['answer_start'].tolist()
    train_context = [context_list[i] for i in train_index]
    valid_context = [context_list[i] for i in valid_index]
    train_answer_start = [answer_start_list[i] for i in train_index]
    valid_answer_start = [answer_start_list[i] for i in valid_index]

    train_dataset = CustomDPRDataset(contexes_list=train_context,
                                     questions_list=train_question,
                                     answer_start_list=train_answer_start)
    valid_dataset = CustomDPRDataset(
        contexes_list=valid_context,
        questions_list=valid_question,
        answer_start_list=valid_answer_start
    )

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=40,
        save_total_limit=5,
        per_device_train_batch_size=1,
        # BATCH SIZE HERE SHOULD ALWAYS BE 1! WE HANDLE IN-BATCH NEGATIVES IN CUSTOM CLASSES.
        gradient_accumulation_steps=12,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        # metric_for_best_model='top_1_accuracy',
        # greater_is_better=True,
        disable_tqdm=False,
        warmup_steps=500,
        do_eval=True,
        fp16=True,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        optim='adamw_hf',
        logging_dir=MODEL_DIR + '/logging',
        run_name='obss-dpr-two-different-encoders',
    )
    CustomDPRConfig.register_for_auto_class()
    HFDPRModel.register_for_auto_class()

    config = CustomDPRConfig()
    model = HFDPRModel(config=config)

    trainer = DPRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[DPRValidationCallback(valid_context, valid_question)],
        data_collator=DPRDataCollator()
    )
    result = trainer.train()
    trainer.save_model(MODEL_DIR)
    print(trainer.state.best_model_checkpoint)

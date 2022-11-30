import itertools

from transformers import DPRContextEncoderTokenizer, \
    DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRQuestionEncoder
import pandas as pd
from torch import nn
import random
from sentence_transformers import util
from sentence_transformers.readers import InputExample
import math
import torch, numpy as np
from typing import List, Dict, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import top_k_accuracy_score, f1_score
import os

#context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
#context_model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

#question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
#question_model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

EPOCH = 50
MODEL_DIR = '/home/secilsen/PycharmProjects/NeuroscienceQA/DPR-model'


class CustomDPRDataLoader:
    def __init__(self, contexes_list,
                 questions_list,
                 batch_size,
                 positive_passages=1
                 ):
        # contexes and questions must be in the same order (contexes_list[i], questions_list[i] pair label should be equal to 1.)
        self.contexes_list = contexes_list
        self.questions_list = questions_list
        if len(self.contexes_list) != len(self.questions_list):
            raise ValueError("Length of context list and question list should be equal!")

        self.data_size = len(self.contexes_list)

        self.batch_size = batch_size
        self.positive_passages = positive_passages

        if self.positive_passages == 1:
            self.loss_strategy = 'negative_ll_positive_loss'
        else:
            self.loss_strategy = 'contrastive_loss'
            self.pos_neg_ratio = 2

    def get_loss_strategy(self):
        return self.loss_strategy

    def __iter__(self):
        zipped = list(zip(self.contexes_list, self.questions_list))
        random.shuffle(zipped)
        self.contexes_list, self.questions_list = zip(*zipped)
        index = 0
        batch = []
        pos_index = 0
        if self.loss_strategy == 'contrastive_loss':
            while index + 1 < self.data_size:
                context = self.contexes_list[index]
                if len(batch) % self.pos_neg_ratio > 0:  # Negative (different) pair
                    index += 1
                    question = self.questions_list[index]
                    label = 0
                else:  # Positive (identical pair)
                    question = self.questions_list[index]
                    label = 1

                index += 1
                batch.append(InputExample(texts=[context, question], label=label))

                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
        else:
            while index + 1 < self.data_size:
                context = self.contexes_list[index]
                if len(batch) > 0:  # Negative (different) pair
                    rand_index = index
                    while index == rand_index:
                        rand_index = random.randrange(self.data_size)
                    question = self.questions_list[rand_index]
                    label = 0
                else:  # Positive (identical pair)
                    question = self.questions_list[index]
                    pos_index = index + 1
                    label = 1

                index += 1
                batch.append(dict(context=context, question=question, label=label))

                if len(batch) >= self.batch_size:
                    yield batch
                    index = pos_index
                    batch = []

    def __len__(self):
        return len(self.contexes_list)
        #return math.floor(len(self.sentences) / (2 * self.batch_size))



class DPRModel(nn.Module):
    def __init__(self,
                 question_model_name='facebook/dpr-question_encoder-single-nq-base',
                 context_model_name='facebook/dpr-ctx_encoder-single-nq-base',
                 freeze_params=12.0):
        super(DPRModel, self).__init__()
        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(context_model_name)
        self.context_model = DPRContextEncoder.from_pretrained(context_model_name)

        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(question_model_name)
        self.question_model = DPRQuestionEncoder.from_pretrained(question_model_name)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.freeze_params = freeze_params
       # self.freeze_layers()

    def cls_pooling(self, model_output):
        return model_output[0][:, 0, :]

    def batch_dot_product(self, context_output, question_output):
        mat1 = torch.unsqueeze(question_output, dim=1)
        mat2 = torch.unsqueeze(context_output, dim=2)
        result = torch.bmm(mat1, mat2)
        result = torch.squeeze(result, dim=1)
        result = torch.squeeze(result, dim=1)
        return result

    def freeze_layers(self):
        num_query_layers = sum(1 for _ in self.context_model.parameters())
        num_passage_layers = sum(1 for _ in self.question_model.parameters())

        for parameters in list(self.query_model.parameters())[:int(self.freeze_params * num_query_layers)]:
            parameters.requires_grad = False

        for parameters in list(self.query_model.parameters())[int(self.freeze_params * num_query_layers):]:
            parameters.requires_grad = True

        for parameters in list(self.passage_model.parameters())[:int(self.freeze_params * num_passage_layers)]:
            parameters.requires_grad = False

        for parameters in list(self.passage_model.parameters())[int(self.freeze_params * num_passage_layers):]:
            parameters.requires_grad = True

    def tokenize(self, batch: Union[List[Dict], Dict]):
        contextes = []
        questions = []
        if isinstance(batch, List):
            for item in batch:
                contextes.append(item['context'])
                questions.append(item['question'])
        else:
            contextes.append(batch['context'])
            questions.append(batch['question'])

        context_tensor = self.context_tokenizer(contextes, padding=True, truncation=True, return_tensors="pt",
                                                add_special_tokens=True)
        question_tensor = self.question_tokenizer(questions, padding=True, truncation=True, return_tensors="pt",
                                                  add_special_tokens=True)
        context_tensor_input_ids = context_tensor['input_ids']
        question_tensor_input_ids = question_tensor['input_ids']
        contexes_attention_mask = context_tensor['attention_mask']
        questions_attention_mask = question_tensor['attention_mask']
        return context_tensor_input_ids, question_tensor_input_ids, contexes_attention_mask, questions_attention_mask

    def forward(self, batch: Union[List[Dict], Dict]):
        context_tensor_input_ids, question_tensor_input_ids, contexes_attention_mask, questions_attention_mask = self.tokenize(
            batch)
        context_model_output = self.context_model(input_ids=context_tensor_input_ids,
                                                  attention_mask=contexes_attention_mask)  # (bsz, hdim)
        question_model_output = self.question_model(input_ids=question_tensor_input_ids,
                                                    attention_mask=questions_attention_mask)
        embeddings_context = context_model_output['pooler_output']
        embeddings_question = question_model_output['pooler_output']

        scores = self.batch_dot_product(embeddings_context, embeddings_question) # self.scale
        return scores
    #   labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)
    #   return (self.cross_entropy_loss(scores, labels) + self.cross_entropy_loss(scores.t(), labels)) / 2


class ContrastiveTensionLossInBatchNegatives(nn.Module):
    def __init__(self, scale: float = 20.0, strategy='contrastive_loss'):
        super(ContrastiveTensionLossInBatchNegatives, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.negative_log_likelihood = nn.NLLLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(scale))
        if strategy == 'contrastive_loss':
            self.loss_func = self.contrastive_tension_loss
        else:
            self.loss_func = self.negative_likelihood_of_positive_passages_v2

    def forward(self, scores, batch):
        labels = [item['label'] for item in batch]
    #    labels = torch.unsqueeze(torch.tensor(labels, dtype=torch.float, device=scores.device), dim=1)
     #   scores *= torch.exp(self.logit_scale)
    #    return (self.cross_entropy_loss(scores, labels) + self.cross_entropy_loss(scores.t(), labels)) / 2
        return self.loss_func(scores, labels)

    def contrastive_tension_loss(self, scores, labels):
        # we need to maximize the loss here!
        loss = 0
        for label, score in zip(labels, scores):
            if label == 0:
                delta = self.m - score
                delta = torch.clamp(delta, min=0.0, max=None)
                loss += torch.mean(torch.pow(delta, 2))
            else:
                loss += torch.mean(torch.pow(score, 2))
        return loss/len(scores)

    def negative_likelihood_of_positive_passages(self, scores, labels): # one positive others negative.
        sum_of_negatives = []
        positive_score = 0.
        for score, label in zip(scores, labels):
            res = torch.clamp(torch.exp(score[0]), min=0.0, max=None)
            if label == 0:
                sum_of_negatives.append(res)
            else:
                positive_score = res# SUPPOSE THAT ONLY ONE POSITIVE PASSAGE IN THE BATCH!!!!!!
        likelihood = positive_score / (positive_score + torch.sum(torch.tensor(sum_of_negatives)))
        return -torch.log(likelihood)

    def negative_likelihood_of_positive_passages_v2(self, scores, labels):
        positive_index = np.where(np.array(labels)==1)
        return torch.logsumexp(scores, dim=0) - (scores[positive_index[0]]*torch.log(torch.exp(torch.tensor(1))))


def calculate_top_k_accuracy(contexes, questions, dprModel, k=20):
    top_k_accuracies = []
    for index_c, context in enumerate(contexes):
        scores_list = []
        labels_list = np.zeros(shape=len(contexes))
        for index_q, question in enumerate(questions):
            if index_c == index_q:
                labels_list[index_c] = 1
            input_dict = dict(context=context, question=question)
            score = dprModel(input_dict)
            scores_list.append(score.detach().cpu().numpy()[0])
        sorted_indexes = sorted(range(len(scores_list)), key=lambda k: scores_list[k], reverse=True)
        predictions = np.zeros(shape=len(contexes))
        for idx in sorted_indexes[:k]:
            predictions[idx] = 1
        acc = top_k_accuracy_score(labels_list, predictions, k=k)
        top_k_accuracies.append(acc)
    mean_accuracy = np.mean(top_k_accuracies)
    return mean_accuracy


if __name__ == '__main__':
    # Data loading
    full_df = pd.read_csv(
        'policyQA_bsbs_sentence.csv', delimiter='|')
    full_df = full_df.drop(labels=['answer'], axis=1)
    train_context, valid_context, train_question, valid_question = train_test_split(full_df['context'].tolist(),
                                                                                    full_df['question'].tolist(),
                                                                                    test_size=0.25, random_state=8)

    dpr_model = DPRModel()
    criterion = ContrastiveTensionLossInBatchNegatives(strategy='negative_ll_positive_loss')
    optimizer = torch.optim.AdamW(dpr_model.parameters(), lr=5e-5, weight_decay=0.01)
    train_dataloader = CustomDPRDataLoader(contexes_list=train_context,
                                           questions_list=train_question,
                                           batch_size=2,
                                           positive_passages=1)
    valid_dataloader = CustomDPRDataLoader(contexes_list=valid_context,
                                           questions_list=valid_question,
                                           batch_size=2,
                                           positive_passages=1)

    torch.autograd.set_detect_anomaly(True)
    for epoch in range(EPOCH):
        epoch_train_loss = []
        epoch_val_loss = []
        eval_accuracy = []
        #  eval_f1 = []
        print(len(train_dataloader))
        for batch_train in train_dataloader:
            optimizer.zero_grad()
            scores = dpr_model(batch_train)
            loss = criterion(scores, batch_train)
            epoch_train_loss.append(loss)
            loss.backward()
            optimizer.step()
        train_loss = torch.mean(torch.stack(epoch_train_loss))
        for batch_valid in valid_dataloader:
            scores = dpr_model(batch_valid)
            loss = criterion(scores, batch_valid)
            epoch_val_loss.append(loss)

        eval_acc = calculate_top_k_accuracy(contexes=valid_context, questions=valid_question, dprModel=dpr_model, k=20)
        valid_loss = torch.mean(torch.stack(epoch_val_loss))

        print(f"Epoch : {epoch + 1} Train Loss: {train_loss} Valid Loss: {valid_loss} Valid Top 20 accuracy: {eval_acc}")
        torch.save(dpr_model.state_dict(), os.path.join(MODEL_DIR, 'epoch-{}.pt'.format(epoch+1)))

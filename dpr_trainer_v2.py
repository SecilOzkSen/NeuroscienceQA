from transformers import DPRContextEncoderTokenizer, \
    DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRQuestionEncoder, PreTrainedModel, PretrainedConfig
import pandas as pd
from torch import nn
import random
import torch, numpy as np
from typing import List, Dict, Union
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments, TrainerCallback
from torch.utils.data import Dataset
import os

os.environ["WANDB_DISABLED"] = "true"

BATCH_SIZE = 12
MODEL_DIR = 'DPR-model'
EPOCH = 50

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


class CustomDPRDataset(Dataset):
    def __init__(self, contexes_list,
                 questions_list,
                 indexes_list,
                 context_tokenizer,
                 question_tokenizer,
                 batch_size = BATCH_SIZE,
                 loss_strategy='negative_ll_positive_loss',
                 ):
        # contexes and questions must be in the same order (contexes_list[i], questions_list[i] pair label should be equal to 1.)
        self.contexes_list = contexes_list
        self.questions_list = questions_list
        self.indexes_list = indexes_list
        if len(self.contexes_list) != len(self.questions_list):
            raise ValueError("Length of context list and question list should be equal!")

        self.data_size = len(self.contexes_list)

        self.batch_size = batch_size
        self.loss_strategy = loss_strategy
        self.pos_neg_ratio = 2
        self.context_tokenizer = context_tokenizer
        self.question_tokenizer = question_tokenizer


    def tokenize(self, batch: Union[List[Dict], Dict]):
        contextes = []
        questions = []
        labels = []
        if isinstance(batch, List):
            for item in batch:
                contextes.append(item['context'])
                questions.append(item['question'])
                labels.append(torch.tensor(item['label'], dtype=torch.int64))
        else:
            contextes.append(batch['context'])
            questions.append(batch['question'])
            labels.append(torch.Tensor(batch['labels']))

        context_tensor = self.context_tokenizer(contextes, padding=True, truncation=True, return_tensors="pt",
                                                add_special_tokens=True)
        question_tensor = self.question_tokenizer(questions, padding=True, truncation=True, return_tensors="pt",
                                                  add_special_tokens=True)
        return [context_tensor, question_tensor], torch.stack(labels)


    def __getitem__(self, index):
        question = self.questions_list[index]
        gt_context = self.contexes_list[index]
        context_index = self.indexes_list[index]
        batch = [dict(question=question, context=gt_context, label=1)]
        if self.loss_strategy == 'contrastive_loss':
            negatives_count = 1
            negatives_index = index + 1
            for i in range(negatives_count):
                context_list_index = 0 if negatives_index >= self.data_size - 1  else negatives_index + 1
                batch.append(dict(question=question, context=self.contexes_list[context_list_index], label=0))
                negatives_index += 1
        else:
            for i in range(self.batch_size - 1):
                negative_context_index = context_index
                context_list_index = index
                while context_index == negative_context_index:
                    negatives_index = index + 1
                    context_list_index = 0 if negatives_index >= self.data_size - 1 else negatives_index
                    negative_context_index = self.indexes_list[context_list_index]

                context = self.contexes_list[context_list_index]
                batch.append(dict(question=question, context=context, label=0))

        tensors, labels = self.tokenize(batch)
        return tensors, labels

    def __len__(self):
        return len(self.contexes_list)

    def shuffle(self):
        zipped = list(zip(self.contexes_list, self.questions_list))
        random.shuffle(zipped)
        self.contexes_list, self.questions_list = zip(*zipped)


class DPRModel(nn.Module):
    def __init__(self,
                 question_model_name='facebook/dpr-question_encoder-single-nq-base',
                 context_model_name='facebook/dpr-ctx_encoder-single-nq-base',
                 freeze_params=12.0):
        super(DPRModel, self).__init__()
        self.freeze_params = freeze_params
        self.question_model = DPRQuestionEncoder.from_pretrained(question_model_name)
        self.context_model = DPRContextEncoder.from_pretrained(context_model_name)
    #    self.freeze_layers(freeze_params)

    def freeze_layers(self, freeze_params):
        num_layers_context = sum(1 for _ in self.context_model.parameters())
        num_layers_question = sum(1 for _ in self.question_model.parameters())

        for parameters in list(self.context_model.parameters())[:int(freeze_params * num_layers_context)]:
            parameters.requires_grad = False

        for parameters in list(self.context_model.parameters())[int(freeze_params * num_layers_context):]:
            parameters.requires_grad = True

        for parameters in list(self.question_model.parameters())[:int(freeze_params * num_layers_question)]:
            parameters.requires_grad = False

        for parameters in list(self.question_model.parameters())[int(freeze_params * num_layers_question):]:
            parameters.requires_grad = True

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
        question_model_output = self.question_model(input_ids = question_tensor['input_ids'],
                                                    attention_mask=question_tensor['attention_mask'])
        embeddings_context = context_model_output['pooler_output']
        embeddings_question = question_model_output['pooler_output']

        scores = self.batch_dot_product(embeddings_context, embeddings_question)  # self.scale
        return scores


class ContrastiveTensionLossInBatchNegatives(nn.Module):
    def __init__(self, scale: float = 20.0, strategy='contrastive_loss'):
        super(ContrastiveTensionLossInBatchNegatives, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.negative_log_likelihood = nn.NLLLoss()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(scale))
        if strategy == 'contrastive_loss':
            self.loss_func = self.contrastive_tension_loss
            self.m = 2.0
        else:
            self.loss_func = self.negative_likelihood_of_positive_passages_v2

    def forward(self, scores, batch):
        labels_list = batch['labels']
        return self.loss_func(scores, labels_list)

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
        return loss / len(scores)

    def negative_likelihood_of_positive_passages(self, scores, labels):  # one positive others negative.
        sum_of_negatives = []
        positive_score = 0.
        for score, label in zip(scores, labels):
            res = torch.clamp(torch.exp(score[0]), min=0.0, max=None)
            if label == 0:
                sum_of_negatives.append(res)
            else:
                positive_score = res  # SUPPOSE THAT ONLY ONE POSITIVE PASSAGE IN THE BATCH!!!!!!
        likelihood = positive_score / (positive_score + torch.sum(torch.tensor(sum_of_negatives)))
        return -torch.log(likelihood)

    def negative_likelihood_of_positive_passages_v2(self, scores, labels:List[torch.Tensor]):
        positive_index = np.where(labels.detach().cpu().numpy() == 1)
        res = torch.logsumexp(scores, dim=0) - (scores[positive_index[0]] * torch.log(torch.exp(torch.tensor(1))))
        return res[0]


class DPRTrainer(Trainer):
    def __init__(self, loss_strategy="contrastive_loss", **kwargs):
        super().__init__(**kwargs)
        self.criterion = ContrastiveTensionLossInBatchNegatives(strategy=loss_strategy)

    def compute_loss(self, model, inputs, return_outputs=False):
        scores = model(inputs)
        loss = self.criterion(scores, inputs)
        return (loss, scores) if return_outputs else loss


def calculate_topk_accuracy(contexes, questions, dprModel, K_range, random_question_size=20):
    acc_dic = {}
    accuracies = np.zeros(shape=(len(questions), K_range))
    for index_q, question in enumerate(questions):
        scores_list = []
        random_indexes = random.sample(range(len(contexes)), random_question_size)
        random_indexes.append(index_q)
        random_indexes = list(set(random_indexes))
        tokenized_question = question_tokenizer(question, padding=True, truncation=True, return_tensors="pt",
                                                add_special_tokens=True)
        tokenized_question.to('cuda')
        gt_labels = np.zeros(shape=len(random_indexes))
        for i, random_index in enumerate(random_indexes):
            if random_index == index_q:
                gt_labels[i] = 1
            tokenized_context = context_tokenizer(contexes[random_index], padding=True, truncation=True, return_tensors="pt",
                                                add_special_tokens=True)
            tokenized_context.to('cuda')
            input_dict = dict(context_tensor=tokenized_context, question_tensor=tokenized_question)
            score = dprModel(input_dict)
            scores_list.append(score.detach().cpu().numpy()[0])
        sorted_indexes = sorted(range(len(scores_list)), key=lambda k: scores_list[k], reverse=True)
        # Top K ranked accuracy calculation
        for _k in range(K_range):
            acc = 0.
            for index in sorted_indexes[:_k+1]:
                if gt_labels[index] == 1:
                    acc = 1.
                    break
            accuracies[index_q, _k] = acc
    for j in range(K_range):
        acc = np.mean(accuracies[:, j])
        acc_dic[f'top_{j+1}_accuracy'] = acc
    return acc_dic

class CustomDataCollator:
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

class CustomDataCollatorForContrastiveLoss:
    def collate_function(self, data):
        ctx_tensors = []
        lbl_tensors = []
        for tensors, labels in data:
            context_tensor, question_tensor = tensors
            ctx_tensors.extend(context_tensor)
            lbl_tensors.extend(question_tensor)

        return dict(context_tensor=ctx_tensors,
              #      context_attention_mask=context_tensor['attention_mask'],
                    question_tensor=lbl_tensors,
             #       question_attention_mask=question_tensor['attention_mask'],
                    labels=labels)
    def __call__(self, data):
        return self.collate_function(data)


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
        accuracy_dict = calculate_topk_accuracy(self.valid_contexes, self.valid_questions, model, K_range=self.K,
                                           random_question_size=self.rand_question_selected)
        state.log_history[-1]['accuracies'] = accuracy_dict


if __name__ == '__main__':
    full_df = pd.read_csv(
        'basecamp.csv', delimiter='|')
    full_df = full_df.drop(labels=['answer'], axis=1)
    indexes = list(range(len(full_df.index)))

    train_index, valid_index, train_question, valid_question = train_test_split(indexes,
                                                                                    full_df['question'].tolist(),
                                                                                    test_size=0.01, random_state=8)
    context_list = full_df['context'].tolist()
    train_context = [context_list[i] for i in train_index]
    valid_context = [context_list[i] for i in valid_index]

    train_dataset = CustomDPRDataset(contexes_list=train_context,
                                     questions_list=train_question,
                                     indexes_list=train_index,
                                     context_tokenizer=context_tokenizer,
                                     question_tokenizer=question_tokenizer,
                                     loss_strategy='negative_ll_positive_loss')
    valid_dataset = CustomDPRDataset(
        contexes_list=valid_context,
        questions_list=valid_question,
        indexes_list=valid_index,
        context_tokenizer=context_tokenizer,
        question_tokenizer=question_tokenizer,
        loss_strategy='negative_ll_positive_loss'
    )

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=50,
        save_total_limit=2,
        per_device_train_batch_size=1, #BATCH SIZE HERE SHOULD ALWAYS BE 1! WE HANDLE IN-BATCH NEGATIVES IN CUSTOM CLASSES.
        gradient_accumulation_steps=12,
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
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
        loss_strategy='negative_ll_positive_loss',
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[DPRValidationCallback(valid_context, valid_question)],
        data_collator=CustomDataCollator(),
    )
    result = trainer.train()
    trainer.save_model(MODEL_DIR)
    print(trainer.state.best_model_checkpoint)


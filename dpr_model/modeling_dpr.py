from transformers import PreTrainedModel, AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from .configuration_dpr import CustomDPRConfig
from typing import Union, List, Dict


class OBSSDPRModel(PreTrainedModel):
    config_class = CustomDPRConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = DPRModel()

    def forward(self, input):
        return self.model(input)


class DPRModel(nn.Module):
    def __init__(self,
                 question_model_name='facebook/contriever-msmarco',
                 context_model_name='facebook/contriever-msmarco'):
        super(DPRModel, self).__init__()
        self.question_model = AutoModel.from_pretrained(question_model_name)
        self.context_model = AutoModel.from_pretrained(context_model_name)

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

    ##FOR CONTRIEVER
    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def forward(self, batch: Union[List[Dict], Dict]):
        context_tensor = batch['context_tensor']
        question_tensor = batch['question_tensor']
        context_model_output = self.context_model(**context_tensor)
        question_model_output = self.question_model(**question_tensor)
        embeddings_context = self.mean_pooling(context_model_output[0], context_tensor['attention_mask'])
        embeddings_question = self.mean_pooling(question_model_output[0], question_tensor['attention_mask'])
        scores = self.batch_dot_product(embeddings_context, embeddings_question)  # self.scale
        return scores

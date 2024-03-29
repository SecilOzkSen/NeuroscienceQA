from transformers import PreTrainedModel, DPRQuestionEncoder, DPRContextEncoder
import torch
import torch.nn as nn
from .configuration_dpr_contrastive_sentence import CustomDPRConfig
from typing import Union, List, Dict

class OBSSDPRContrastiveSentence(PreTrainedModel):
    config_class = CustomDPRConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = DPRModel()

    def forward(self, input):
        return self.model(input)

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
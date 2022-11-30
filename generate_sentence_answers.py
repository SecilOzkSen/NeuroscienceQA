from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
import torch
from torch.autograd import Variable
import spacy

MODEL_PATH = "/home/secilsen/PycharmProjects/NeuroscienceQA/roberta-large-asnq-policyqa/checkpoint-2008"
TOKENIZER_PATH = "/home/secilsen/PycharmProjects/NeuroscienceQA/roberta-base-asnq"
spacy_nlp = spacy.load('en_core_web_sm')

def get_sentences_from_context(context):
    doc = spacy_nlp(context)
    sentences = []
    for sentence in doc.sents:
        sentences.append(sentence.text)
    return sentences

def sentence_only():
    question = "Do these policies apply to me?"
    sentence = "These current privacy policies and the privacy practices described herein apply to personal information previously collected from you by Barnes & Noble, Inc. and each of its subsidiaries, as well as personal information that they may collect from you in the future."
    tokenized = tokenizer(f"<s> {question} </s> {sentence}", padding=True, truncation=True, return_tensors='pt')
    output = model(**tokenized)
    soft_outputs = torch.nn.functional.sigmoid(output[0])
    t = Variable(torch.Tensor([0.6 ]))  # threshold
    out = (soft_outputs[0] > t).float() * 1
    res = torch.argmax(out, dim=-1)
    prob = soft_outputs[:, res].flatten().cpu().detach().numpy()
    if out[1] == 1:
        print("Answer")

    res_list = res.cpu().detach().numpy()
    print(res_list[0])
    print(prob[0])

def context_sentences():
    question = "What does the MFA stand for?"
    context = '''
The GW Medical Faculty Associates (MFA) is committed to respecting your privacy. In general, you can visit The MFA on the Web without identifying yourself or revealing any personal information. In some areas, however, you may choose services that require you to provide us with information by which you can be identified. Once any personally identifiable information is received, you can be assured that it will only be used to support your relationship with the MFA.     
'''
    sentences = get_sentences_from_context(context.strip())
    for sentence in sentences:
        tokenized = tokenizer(f"<s> {question} </s> {sentence} </s>", padding=True, truncation=True, return_tensors='pt')
        output = model(**tokenized)
        soft_outputs = torch.nn.functional.sigmoid(output[0])
        t = Variable(torch.Tensor([0.9]))  # threshold
        out = (soft_outputs[0] > t) * 1.
        out = out.flatten().cpu().detach().numpy()
     #   res = torch.argmax(out, dim=-1)
        prob = soft_outputs[:, 1].flatten().cpu().detach().numpy()
        if out[1] == 1:
            print(f"ANSWER with PROB {prob} : {sentence}")


if __name__ == '__main__':
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = RobertaTokenizer.from_pretrained(TOKENIZER_PATH)
    context_sentences()


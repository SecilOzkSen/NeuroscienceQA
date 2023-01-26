import copy
import pickle

import pandas as pd
from transformers import AutoModel, AutoTokenizer
import json
from sentence_transformers import SentenceTransformer
import spacy

dpr_model = AutoModel.from_pretrained("secilozksen/dpr_basecamp_contriever_contrastive",
                                      use_auth_token="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", trust_remote_code=True)
dpr_context_model = copy.deepcopy(dpr_model.model.context_model)
del dpr_model
dpr_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")


# nlp = spacy.load("en_core_web_sm")

def load_pickle_file(path):
    with open(path, "rb") as fIn:
        cache_data = pickle.load(fIn)
        corpus_contexes = cache_data['contexes']
        corpus_embeddings = cache_data['embeddings']
    return corpus_contexes, corpus_embeddings


def load_squad_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = data['data']
    contexes_list = []
    for paragraphs in data:
        for paragraph in paragraphs['paragraphs']:
            context = paragraph['context'].lstrip('.')
            context = context.strip(' ')
            contexes_list.append(context)
    return contexes_list


def load_csv(path):
    data = pd.read_csv(path, delimiter='|', encoding='utf-8')
    data = data.drop_duplicates(subset=['context_id'], keep='first')
    return data


def create_context_embeddings(contexes):
    embeddings = []
    contexes_list = []
    for context in contexes:
        if context in contexes_list:
            continue
        contexes_list.append(context)
        tokenized = dpr_tokenizer(context, padding=True, truncation=True, return_tensors="pt")
        context_embeddings = dpr_context_model(**tokenized)
        # pooler_outputs = context_embeddings['pooler_output']
        embeddings_context = mean_pooling(context_embeddings[0], tokenized['attention_mask'])
        embeddings.append(embeddings_context)

    with open('basecamp-dpr-contriever-embeddings.pkl', "wb") as fIn:
        pickle.dump({'contexes': contexes, 'embeddings': embeddings}, fIn)


def sentence_transformers_create_context_embeddings(contexes):
    bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    bi_encoder.max_seq_length = 500
    with open('data/st-context-embeddings.pkl', "wb") as fIn:
        context_embeddings = bi_encoder.encode(contexes, convert_to_tensor=True, show_progress_bar=True)
        pickle.dump({'contexes': contexes, 'embeddings': context_embeddings}, fIn)


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


if __name__ == '__main__':
    #  corpus_contexes, _ = load_pickle_file("context-embeddings.pkl")
    #    corpus_contexes = load_squad_data('/home/secilsen/PycharmProjects/SquadOperations/basecamp_bsbs.json')
    corpus_df = load_csv('data/basecamp.csv')
    corpus_contexes = corpus_df['context'].tolist()
    corpus_ids = corpus_df['context_id'].tolist()
    create_context_embeddings(corpus_contexes)

#  sentence_transformers_create_context_embeddings(corpus_contexes, corpus_ids)

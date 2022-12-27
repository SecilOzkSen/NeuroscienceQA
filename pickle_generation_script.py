import copy
import pickle
from transformers import AutoModel, AutoTokenizer
import json
from sentence_transformers import SentenceTransformer

dpr_model = AutoModel.from_pretrained("secilozksen/dpr_policyqa", use_auth_token="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", trust_remote_code=True)
dpr_context_model = copy.deepcopy(dpr_model.model.context_model)
del dpr_model
dpr_tokenizer = AutoTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

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


def create_context_embeddings(contexes):
    embeddings = []
    for context in contexes:
        tokenized = dpr_tokenizer(context, padding=True, truncation=True, return_tensors="pt",
                                      add_special_tokens=True)
        context_embeddings = dpr_context_model(**tokenized)
        pooler_outputs = context_embeddings['pooler_output']
        embeddings.append(pooler_outputs)

    with open('basecamp-context-embeddings.pkl', "wb") as fIn:
        pickle.dump({'contexes': contexes, 'embeddings': embeddings}, fIn)

def sentence_transformers_create_context_embeddings(contexes):
    bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    bi_encoder.max_seq_length = 500
    with open('st-context-embeddings.pkl', "wb") as fIn:
        context_embeddings = bi_encoder.encode(contexes, convert_to_tensor=True, show_progress_bar=True)
        pickle.dump({'contexes': contexes, 'embeddings': context_embeddings}, fIn)

if __name__ == '__main__':
  #  corpus_contexes, _ = load_pickle_file("context-embeddings.pkl")
    corpus_contexes = load_squad_data('basecamp_squad.json')
    sentence_transformers_create_context_embeddings(corpus_contexes)

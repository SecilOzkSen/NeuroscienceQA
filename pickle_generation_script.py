import copy
import pickle
from transformers import AutoModel, AutoTokenizer

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

def create_context_embeddings(contexes):
    embeddings = []
    for context in contexes:
        tokenized = dpr_tokenizer(context, padding=True, truncation=True, return_tensors="pt",
                                      add_special_tokens=True)
        context_embeddings = dpr_context_model(**tokenized)
        pooler_outputs = context_embeddings['pooler_output']
        embeddings.append(pooler_outputs)

    with open('dpr-context-embeddings.pkl', "wb") as fIn:
        pickle.dump({'contexes': contexes, 'embeddings': embeddings}, fIn)

if __name__ == '__main__':
    corpus_contexes, _ = load_pickle_file("/home/secilsen/PycharmProjects/semanticSearchDemo/context-embeddings.pkl")
    create_context_embeddings(corpus_contexes)

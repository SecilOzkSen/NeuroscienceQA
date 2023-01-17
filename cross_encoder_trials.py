from sentence_transformers import util, SentenceTransformer, CrossEncoder
from transformers import RobertaModel, AutoModel, RobertaTokenizer, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
import pickle
import torch


TRAINED_MODEL_PATH = "secilozksen/dpr_policyqa"

def load_bi_encoder():
    bi_encoder = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    bi_encoder.max_seq_length = 500  # Truncate long passages to 256 tokens
    return bi_encoder

def load_trained_cross_encoder():
    model = CrossEncoder("/home/secilsen/PycharmProjects/NeuroscienceQA/cross-encoder-model-contrastive")
    return model

def load_trained_dpr_model():
    dpr_model = AutoModel.from_pretrained(TRAINED_MODEL_PATH, use_auth_token="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", trust_remote_code=True)
    return dpr_model

def load_cross_encoder_tokenizer():
    cross_encoder_tokenizer = RobertaTokenizer.from_pretrained("/home/secilsen/PycharmProjects/NeuroscienceQA/cross-encoder-model")
    return cross_encoder_tokenizer

def load_dpr_question_tokenizers():
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    return question_tokenizer

def load_dpr_context_tokenizers():
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    return context_tokenizer


def load_pickle_file(path):
    with open(path, "rb") as fIn:
        cache_data = pickle.load(fIn)
        corpus_contexes = cache_data['contexes']
        corpus_embeddings = cache_data['embeddings']
    return corpus_contexes, corpus_embeddings

def retrieve(question, corpus_embeddings):
    # Semantic Search (Retrieve)
    question_embedding = bi_encoder.encode(question, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=100)
    if len(hits) == 0:
        return []
    hits = hits[0]
    return hits

def retrieve_with_dpr_embeddings(question, corpus_embeddings):
    # Semantic Search (Retrieve)
    question_tokens = question_tokenizer(question, padding=True, truncation=True, return_tensors="pt",
                                            add_special_tokens=True)

    question_embedding = dpr_model.model.question_model(**question_tokens)['pooler_output']
    question_embedding = torch.squeeze(question_embedding, dim=0)
    corpus_embeddings = torch.stack(corpus_embeddings)
    corpus_embeddings = torch.squeeze(corpus_embeddings, dim=1)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=100, score_function=util.dot_score)
    if len(hits) == 0:
        return []
    hits = hits[0]
    return hits, question_embedding

def rerank_with_trained_cross_encoder(hits, question, contexes):
    # Rerank - score all retrieved passages with cross-encoder
    cross_inp = [(question, contexes[hit['corpus_id']]) for hit in hits]
    cross_scores = trained_cross_encoder.predict(cross_inp)
    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    # Output of top-5 hits from re-ranker
    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    top_5_contexes = []
    top_5_scores = []
    for hit in hits[0:5]:
        top_5_contexes.append(contexes[hit['corpus_id']])
        top_5_scores.append(hit['cross-score'])
    return top_5_contexes, top_5_scores

def batch_dot_product(context_output, question_output):
    mat1 = torch.squeeze(question_output, 0)
    mat2 = torch.squeeze(context_output, 0)
    result = torch.dot(mat1, mat2)
    return result

def DPR_only(question_embedding, selected_contexes, selected_embeddings):
    scores = []
#    tokenized_question = question_tokenizer(question, padding=True, truncation=True, return_tensors="pt",
#                                            add_special_tokens=True)
#    question_output = dpr_model.model.question_model(**tokenized_question)
#    question_output = question_output['pooler_output']
    for context_embedding in selected_embeddings:
        score = batch_dot_product(context_embedding, question_embedding)
        scores.append(score.detach().cpu())

    scores_index = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
    contexes_list = []
    scores_final = []
    for i, idx in enumerate(scores_index[:5]):
        scores_final.append(scores[idx])
        contexes_list.append(selected_contexes[idx])
    return scores_final, contexes_list


def DPR_pipeline(question, contexes):
    scores = []
    tokenized_question = question_tokenizer(question, padding=True, truncation=True, return_tensors="pt",
                                            add_special_tokens=True)
    for context in contexes:
        tokenized_contexes = context_tokenizer(context, padding=True, truncation=True, return_tensors="pt",
                                            add_special_tokens=True)
        score = dpr_model(dict(context_tensor=tokenized_contexes, question_tensor=tokenized_question))
        scores.append(score.detach().cpu())

    scores_index = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
    contexes_list = []
    scores_final = []
    for i, idx in enumerate(scores_index[:5]):
        scores_final.append(scores[idx])
        contexes_list.append(contexes[idx])
    return scores_final, contexes_list


def rerank_with_DPR(hits, question_embedding, contexes, contexes_embeddings):
    # Rerank - score all retrieved passages with cross-encoder
    selected_contexes = [contexes[hit['corpus_id']] for hit in hits]
    selected_embeddings = [contexes_embeddings[hit['corpus_id']] for hit in hits]
    top_5_scores, top_5_contexes = DPR_only(question_embedding, selected_contexes, selected_embeddings)
    return top_5_contexes, top_5_scores


bi_encoder = load_bi_encoder()
trained_cross_encoder = load_trained_cross_encoder()
dpr_model = load_trained_dpr_model()
cross_encoder_tokenizer = load_cross_encoder_tokenizer()
question_tokenizer = load_dpr_question_tokenizers()
context_tokenizer = load_dpr_context_tokenizers()

dpr_corpus_contexes, dpr_corpus_embeddings = load_pickle_file('/home/secilsen/PycharmProjects/semanticSearchDemo/custom-dpr-context-embeddings.pkl')
corpus_contexes, corpus_embeddings = load_pickle_file('/data/st-context-embeddings.pkl')
#question = 'How many times can you make a request for an amendment to a health record?'

question = 'Hello, how are you?'

# Retrieve-Rerank with DPR
#hits = retrieve_with_dpr_embeddings(question, dpr_corpus_embeddings)
#top_5_contexes, top_5_scores = rerank_with_DPR(hits, question, dpr_corpus_contexes, dpr_corpus_embeddings)

# Retrieve-Rerank with trained cross encoder
#hits = retrieve(question, corpus_embeddings)
#top_5_contexes, top_5_scores = rerank_with_trained_cross_encoder(hits, question, corpus_contexes)

# DPR only:
#top_5_scores, top_5_contexes = DPR_only(question, dpr_corpus_contexes, dpr_corpus_embeddings)

# Retrieve-Rerank with DPR
#hits, question_embedding = retrieve_with_dpr_embeddings(question, dpr_corpus_embeddings)
#top_5_contexes, top_5_scores = rerank_with_DPR(hits, question_embedding, dpr_corpus_contexes, dpr_corpus_embeddings)
#for context, score in zip(top_5_contexes, top_5_scores):
#print(f"Score - {score}")
#print(context)

if __name__ == '__main__':
    # Retrieve-Rerank with trained cross encoder
    # DPR only:
    # top_5_scores, top_5_contexes = DPR_only(question, dpr_corpus_contexes, dpr_corpus_embeddings)
    hits = retrieve(question, corpus_embeddings)
    top_5_contexes, top_5_scores = rerank_with_trained_cross_encoder(hits, question, corpus_contexes)
    for context, score in zip(top_5_contexes, top_5_scores):
        print(f"score: {score} \n {context}")



# Retrieve-Rerank with cross-encoder(trained)








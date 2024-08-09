from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import pickle
import os
import pandas as pd
import numpy as np


    
# -------------------- #
#    Model Function    #
# -------------------- #
# Helper: Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def model_fn(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    return model, tokenizer


def model_encode(model, tokenizer, text):
    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform pooling
    text_emb = mean_pooling(model_output, encoded_input['attention_mask'])
    text_emb = F.normalize(text_emb, p=2, dim=1)
    return text_emb


# ----------------------- #
#      Tag TopK Search    #
# ----------------------- #
LOCAL_TAG_INDEX_FOLDER = "local_tag_emb_index"

LOCAL_TAG_INDEX = [
    "dining-tag-vector", 
    "experience-tag-vector", 
    "accommodation-tag-vector",
    "accommodation-brand-tag-vector",
    "tui-accommodation-tag-vector",
    "ids-accommodation-brand-tag-vector",
]


def load_index(index):
    with open(os.path.join(LOCAL_TAG_INDEX_FOLDER, f"{index}.pkl"), 'rb') as f:
        data = pickle.load(f)
    return { "tags": data["tags"], "embeddings": data["embeddings"] }


INDEX2DATA = { idx:load_index(idx) for idx in LOCAL_TAG_INDEX }


def retrieve_topk_tags(query_emb, query_index, topk=5):
    assert query_emb.shape  == (1,384)
    index_data = INDEX2DATA[query_index]
    tag_list, tag_embeddings = index_data['tags'], index_data['embeddings']
    similarities = F.cosine_similarity(query_emb, tag_embeddings, dim=1)
    # Sort indices based on similarity in descending order
    sorted_indices = torch.argsort(similarities, descending=True).tolist()
    topk_tags = [tag_list[idx] for idx in sorted_indices[:topk]]
    return topk_tags


def predict_topk_tags(query, query_index, query_topK, model_and_tokenizer):
    # destruct model and tokenizer
    model, tokenizer = model_and_tokenizer
    query_emb = model_encode(model, tokenizer, query)
    topK_tags = retrieve_topk_tags(query_emb, query_index, query_topK)
    # return dictonary, which will be json serializable
    return {"topK_tags": topK_tags}


# --------------------------- #
#      Tui POI TopK Search    #
# --------------------------- #
TUI_POI_DF = pd.read_parquet(f"./poi_desc_emb/tui-musement-backup-accommodation-embedded.parquet")
TUI_POI_EMB = torch.Tensor(np.vstack(TUI_POI_DF['all-MiniLM-L12-v2']))

def retrieve_topk_pois(query_emb, topk=5):
    assert query_emb.shape == (1,384), f'expected to get query_emb shape (1,384), but get {query_emb.shape}'
    similarities = F.cosine_similarity(query_emb, TUI_POI_EMB, dim=1)
    # Sort indices based on similarity in descending order
    sorted_indices = torch.argsort(similarities, descending=True).tolist()
    return TUI_POI_DF.iloc[sorted_indices[:topk]]


def predict_topk_pois(query, query_topK, model_and_tokenizer, query_columns=None):
    # destruct model and tokenizer
    model, tokenizer = model_and_tokenizer
    query_emb = model_encode(model, tokenizer, query)
    topK_pois = retrieve_topk_pois(query_emb, query_topK)
    
    if query_columns:
        result = topK_pois[query_columns].to_dict(orient='records')
    else:
        result = topK_pois.to_dict(orient='records')

    return {"topK_pois": result}
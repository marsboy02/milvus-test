from transformers import BertModel, BertTokenizer
import torch

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)


def get_column_embedding(column_name):
    inputs = tokenizer(column_name, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach()


def get_entry_embedding(column_names):
    embeddings = [get_column_embedding(col) for col in column_names]
    entry_embedding = torch.stack(embeddings).mean(dim=0)
    return entry_embedding
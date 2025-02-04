import requests
from datasets import load_dataset
import chromadb
from chromadb.utils import embedding_functions
import torch
import numpy as np
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

max_seq_length = 1024
input_ids = []

LoCoV1_dataset = load_dataset("hazyresearch/LoCoV1-Documents", split = "test")
first_row = LoCoV1_dataset[0]
columns = list(first_row.keys())
print(columns)
text_list = [item['passage'] for item in LoCoV1_dataset]
print("First few passages:", text_list[:3])
embedding_file_path = "embeddings.pkl"
embeddings_list = []
batch_size = 64


model = AutoModelForSequenceClassification.from_pretrained(
  "togethercomputer/m2-bert-80M-32k-retrieval",
  trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
  "bert-base-uncased",
  model_max_length=max_seq_length
)



input_ids = tokenizer(
  text_list,
  return_tensors="pt",
  padding="max_length",
  return_token_type_ids=False,
  truncation=True,
  max_length=max_seq_length
)
input_ids = []
batch_size = 1
embedding_file_path = "embeddings.pkl"

with open(embedding_file_path, 'wb') as f, torch.no_grad():
    with tqdm(total=len(text_list), desc="Generating Embeddings") as pbar:
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i + batch_size]
            batch_input_ids = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                return_token_type_ids=False
            )
            
            batch_input_ids = {k: v.to(model.device) for k, v in batch_input_ids.items()}
            
            outputs = model(**batch_input_ids)
            embeddings = outputs['sentence_embedding'].cpu()
            
            # Write each batch's embeddings to file incrementally
            pickle.dump(embeddings, f)
            
            pbar.update(len(batch))

print(embeddings_list)
print(f"Embeddings saved to {embedding_file_path}")


import requests
from datasets import load_dataset
import chromadb
from chromadb.utils import embedding_functions
import torch
import numpy as np
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LoCoV1_dataset = load_dataset("hazyresearch/LoCoV1-Documents", split = "test")
pids = [item['pid'] for item in LoCoV1_dataset]

embedding_file_path = "embeddings.pkl"
embeddings_list = []
# with open(embedding_file_path, 'rb') as f:
#     embeddings_list = pickle.load(f)
with open(embedding_file_path, 'rb') as f:
    while True:
        try:
            # Read each batch of embeddings
            embeddings = pickle.load(f)
            embeddings_list.append(embeddings)
        except EOFError:
            # End of file reached
            break
        
flattened_list = [item for tensor in embeddings_list for sublist in tensor.tolist() for item in sublist]
# print(flattened_list[0])
# print(flattened_list[1])
# print(flattened_list[2])
# print(type(flattened_list))
# print(type(flattened_list[0]))
# print(len(flattened_list))
# print(flattened_list[:3])
# print(pids[0])
# print(pids[1])
# print(pids[2])
# print(type(pids))
# print(type(pids[0]))
# print(len(pids))
# print(pids[:3])
# print(embeddings_list)
embedding_dim = 768
reshaped_embeddings = np.array(flattened_list).reshape(-1, embedding_dim).tolist()
# print(len(reshaped_embeddings))

# print(f"Embedding file loaded, length: {len(embeddings_list)}")
# assert len(embeddings_list) == len(pids), "Mismatch"

# embeddings_list_short = embeddings_list[3]

client = chromadb.Client()

collection = client.create_collection(name="LoCoV1_collection")

print("Creating collection")
collection.add(
    ids=pids,
    embeddings=reshaped_embeddings,
)

model = AutoModelForSequenceClassification.from_pretrained(
  "togethercomputer/m2-bert-80M-32k-retrieval",
  trust_remote_code=True
)

max_seq_length = 1024
tokenizer = AutoTokenizer.from_pretrained(
  "bert-base-uncased",
  model_max_length=max_seq_length
)

# print("Running test query")
while True:
  query_text = input("Enter your query: ")

  
  query_input_ids = tokenizer(
    query_text,
    return_tensors="pt",
    padding="max_length",
    return_token_type_ids=False,
    truncation=True,
    max_length=max_seq_length
  )

  with torch.no_grad():
      query_input_ids = {k: v.to(model.device) for k, v in query_input_ids.items()}
      query_outputs = model(**query_input_ids)
      query_embedding = query_outputs['sentence_embedding'].cpu().numpy().flatten()

  query_embedding = query_embedding.tolist()

  query_result = collection.query(
      query_embeddings=[query_embedding],
      n_results=5
  )

  print("Query results:")
  # print(query_result["ids"])
  document_ids = query_result["ids"][0]
  similarity_scores = query_result["distances"][0]
  print(document_ids)

  for doc_id, score in zip(document_ids, similarity_scores):
      print(f"ID: {doc_id}, Similarity: {score}")
      result = next((item for item in LoCoV1_dataset if item['pid'] == doc_id), None)
      passage = result.get("passage", "No passage column found")
      print("Passage:")
      print(passage[:200])
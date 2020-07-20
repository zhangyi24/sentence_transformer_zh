"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""
import os
import csv
import logging
import pickle
import time

import scipy.spatial
import numpy as np

from sentence_transformers import SentenceTransformer

print('loading model...')
model = SentenceTransformer('../training_transformers/output/training_nli_bert-base-uncased-2020-07-20_10-46-19')

datasets_dir = '../datasets/intent/zj'
trainset_path = os.path.join(datasets_dir, 'train.tsv')
devset_path = os.path.join(datasets_dir, 'dev.tsv')
testset_path = os.path.join(datasets_dir, 'test.tsv')
embedding_cache_path = 'embedding.pkl'

def get_dataset(dataset_path):
    dataset = {}
    with open(dataset_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            dataset[row[1]] = row[0]
    return dataset


trainset = get_dataset(trainset_path)
devset = get_dataset(devset_path)
testset = get_dataset(testset_path)

corpus = list(trainset.keys())

if not os.path.exists(embedding_cache_path):
    embeddings_dict = {}
    print('encoding corpus...')
    corpus_embeddings = model.encode(corpus)
    for sentence, embedding in zip(corpus, corpus_embeddings):
        embeddings_dict[sentence] = embedding
    print('dumping corpus embeddings...')
    with open(embedding_cache_path, 'wb') as f:
        pickle.dump(embeddings_dict, f)
else:
    print('loading corpus embeddings...')
    with open(embedding_cache_path, 'rb') as f:
        embeddings_dict = pickle.load(f)
        corpus_embeddings = list(embeddings_dict.values())


# Query sentences:
print('encoding queries...')
queries = list(testset.keys())
query_embeddings = model.encode(queries)

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
print('KNN...')
distances_matrix = scipy.spatial.distance.cdist(query_embeddings, corpus_embeddings, "cosine")
pred_idx = np.argmax(distances_matrix, axis=1)
true_num = 0
for query, pred_id in zip(queries, pred_idx):
    nearest_sent = corpus[int(pred_id)]
    if testset[query] == trainset[nearest_sent]:
        true_num += 1
print(true_num / len(queries))


# Query sentences:
print('encoding queries...')
queries = list(devset.keys())
query_embeddings = model.encode(queries)

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
print('KNN...')
distances_matrix = scipy.spatial.distance.cdist(query_embeddings, corpus_embeddings, "cosine")
pred_idx = np.argmax(distances_matrix, axis=1)
true_num = 0
for query, pred_id in zip(queries, pred_idx):
    nearest_sent = corpus[int(pred_id)]
    if devset[query] == trainset[nearest_sent]:
        true_num += 1
print(true_num / len(queries))



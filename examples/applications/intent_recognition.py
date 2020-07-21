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
model = SentenceTransformer('../training_transformers/output/training_cmnli_bert-base-chinese-2020-07-20_17-35-51')

datasets_dir = '../datasets/intent/zj'
trainset_path = os.path.join(datasets_dir, 'train.tsv')
devset_path = os.path.join(datasets_dir, 'dev.tsv')
testset_path = os.path.join(datasets_dir, 'test.tsv')
embedding_cache_path = 'embeddings.pkl'

def get_dataset(dataset_path):
    sents = []
    labels = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row_id, row in enumerate(reader):
            if row_id == 0:
                continue
            labels.append(row[0])
            sents.append(row[1])
    return sents, labels


train_sents, train_labels = get_dataset(trainset_path)
dev_sents, dev_labels = get_dataset(devset_path)
test_sents, test_labels = get_dataset(testset_path)

corpus = train_sents
if not os.path.exists(embedding_cache_path):
    print('encoding corpus...')
    corpus_embeddings = model.encode(corpus)
    print('dumping corpus embeddings...')
    with open(embedding_cache_path, 'wb') as f:
        pickle.dump(corpus_embeddings, f)
else:
    print('loading corpus embeddings...')
    with open(embedding_cache_path, 'rb') as f:
        corpus_embeddings = pickle.load(f)



# # Query sentences:
# print('encoding queries...')
# queries = ['没钱没钱']
# query_embeddings = model.encode(queries)
#
# # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
# print('KNN...')
# for query, query_embedding in zip(queries, query_embeddings):
#     distances = scipy.spatial.distance.cdist([query_embedding], train_sents_embeddings, "cosine")[0]
#
#     results = zip(range(len(distances)), distances)
#     results = sorted(results, key=lambda x: x[1])
#
#     print("\n\n======================\n\n")
#     print("Query:", query)
#     print("\nTop 5 most similar sentences in corpus:")
#
#     for idx, distance in results[0:5]:
#         print(corpus[idx].strip(), "(Score: %.4f)" % (1-distance))

# Query sentences:
print('encoding queries...')
queries = test_sents
query_embeddings = model.encode(queries)
# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
print('KNN...')
distances_matrix = scipy.spatial.distance.cdist(query_embeddings, corpus_embeddings, "cosine")
pred_idx = np.argmin(distances_matrix, axis=1)
true_num = 0
for query_id, query in enumerate(queries):
    pred_id = pred_idx[query_id]
    nearest_sent = corpus[pred_id]
    if test_labels[query_id] == train_labels[pred_id]:
        true_num += 1
    # print(query, nearest_sent, test_labels[query_id], train_labels[pred_id])
print(true_num / len(queries))


# Query sentences:
print('encoding queries...')
queries = dev_sents
query_embeddings = model.encode(queries)
# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
print('KNN...')
distances_matrix = scipy.spatial.distance.cdist(query_embeddings, corpus_embeddings, "cosine")
pred_idx = np.argmin(distances_matrix, axis=1)
true_num = 0
for query_id, query in enumerate(queries):
    pred_id = pred_idx[query_id]
    nearest_sent = corpus[pred_id]
    if dev_labels[query_id] == train_labels[pred_id]:
        true_num += 1
    # print(query, nearest_sent, dev_labels[query_id], train_labels[pred_id])
print(true_num / len(queries))




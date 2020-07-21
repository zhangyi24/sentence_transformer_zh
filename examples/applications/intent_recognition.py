"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))

import csv
import logging
import pickle
import time
import argparse

import scipy.spatial
import numpy as np

from sentence_transformers import SentenceTransformer


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


def evaluate(corpus, corpus_embeddings, queries, labels):
    print('encoding queries...')
    query_embeddings = model.encode(queries)
    # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
    print('KNN...')
    distances_matrix = scipy.spatial.distance.cdist(query_embeddings, corpus_embeddings, "cosine")
    pred_idx = np.argmin(distances_matrix, axis=1)
    true_num = 0
    for query_id, query in enumerate(queries):
        pred_id = pred_idx[query_id]
        if labels[query_id] == train_labels[pred_id]:
            true_num += 1
    return true_num / len(queries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", default="bert-base-chinese", type=str)
    args = parser.parse_args()
    print('loading model: %s...' % args.model_name)
    model = SentenceTransformer(args.model_name)

    datasets_dir = '../datasets/intent/zj'
    trainset_path = os.path.join(datasets_dir, 'train.tsv')
    devset_path = os.path.join(datasets_dir, 'dev.tsv')
    testset_path = os.path.join(datasets_dir, 'test.tsv')
    train_sents, train_labels = get_dataset(trainset_path)
    dev_sents, dev_labels = get_dataset(devset_path)
    test_sents, test_labels = get_dataset(testset_path)

    corpus = train_sents
    print('encoding corpus...')
    corpus_embeddings = model.encode(corpus)

    print('\nevalueting on devset:')
    acc = evaluate(corpus, corpus_embeddings, dev_sents, dev_labels)
    print('acc: {:.4%}'.format(acc))

    print('\nevalueting on testset:')
    acc = evaluate(corpus, corpus_embeddings, test_sents, test_labels)
    print('acc: {:.4%}'.format(acc))

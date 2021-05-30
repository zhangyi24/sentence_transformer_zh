"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the CMNLI dataset
with softmax loss function. At every 1000 training steps, the model is evaluated on the
STS benchmark dataset

Usage:
python training_cmnli.py

OR
python training_cmnli.py pretrained_transformer_model_name
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
import logging
from datetime import datetime
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default='bert-base-chinese', type=str, help="model name or model dir")
    parser.add_argument("--num_epochs", "-e", default=1, type=int, help="number of epochs")
    parser.add_argument("--pooling", "-p", default='mean', type=str, help="pooling method", choices=["mean", "cls", "max"])
    parser.add_argument("--batch_size", "-b", default=16, type=int, help="batch size")
    args = parser.parse_args()
    model_name = args.model
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    # Read the dataset

    cmnli_reader = CMNLIDataReader('../datasets/CMNLI')
    csts_reader = CSTSBenchmarkDataReader('../datasets/CSTS-B')
    train_num_labels = cmnli_reader.get_num_labels()
    model_save_path = 'output/%s-%s-cmnli-%s' % (model_name.rstrip("/").split("/")[-1], args.pooling, datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"))

    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    transformers_cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'cache', 'transformers')
    word_embedding_model = models.Transformer(model_name, cache_dir=transformers_cache_dir)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=args.pooling == 'mean',
                                   pooling_mode_cls_token=args.pooling == 'cls',
                                   pooling_mode_max_tokens=args.pooling == 'max')

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Convert the dataset to a DataLoader ready for training
    logging.info("Read CMNLI train dataset")
    train_data = SentencesDataset(cmnli_reader.get_examples('train'), model=model)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                    num_labels=train_num_labels)

    logging.info("Read CSTSbenchmark dev dataset")
    dev_data = SentencesDataset(examples=csts_reader.get_examples('cnsd-sts-dev.txt'), model=model)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
    evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs / batch_size * 0.1)  # 10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              output_path=model_save_path
              )

    ##############################################################################
    #
    # Load the stored model and evaluate its performance on STS benchmark dataset
    #
    ##############################################################################

    model = SentenceTransformer(model_save_path)
    test_data = SentencesDataset(examples=csts_reader.get_examples("cnsd-sts-test.txt"), model=model)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    evaluator = EmbeddingSimilarityEvaluator(test_dataloader)

    model.evaluate(evaluator)

"""
This examples trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) for the CQQP from scratch. It generates sentence embeddings
that can be compared using cosine-similarity to measure the similarity.

Usage:
python training_cqqp.py

OR
python training_cqqp.py pretrained_transformer_model_name
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))

import argparse
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import CQQPDataReader
import logging
from datetime import datetime
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default='bert-base-chinese', type=str, help="model name or model dir")
    parser.add_argument("--num_epochs", "-e", default=10, type=int, help="number of epochs")
    parser.add_argument("--pooling", "-p", default='mean', type=str, help="pooling method",
                        choices=["mean", "cls", "max"])
    parser.add_argument("--batch_size", "-b", default=32, type=int, help="batch_size")
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

    model_save_path = 'output/%s-%s-qqp-%s' % (
    model_name.rstrip("/").split("/")[-1], args.pooling, datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S"))
    cqqp_reader = CQQPDataReader('../datasets/CQQP', normalize_scores=True)

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
    logging.info("Read CQQP train dataset")
    train_data = SentencesDataset(cqqp_reader.get_examples("cqqp-train.csv"), model)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)


    logging.info("Read CQQP dev dataset")
    dev_data = SentencesDataset(examples=cqqp_reader.get_examples("cqqp-dev.csv"), model=model)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=batch_size)
    evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)


    # Configure the training. We skip evaluation in this example
    warmup_steps = math.ceil(len(train_data)*num_epochs/batch_size*0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))


    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              output_path=model_save_path)


    ##############################################################################
    #
    # Load the stored model and evaluate its performance on CQQP dataset
    #
    ##############################################################################

    model = SentenceTransformer(model_save_path)
    test_data = SentencesDataset(examples=cqqp_reader.get_examples("cqqp-test.csv"), model=model)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    evaluator = EmbeddingSimilarityEvaluator(test_dataloader)
    model.evaluate(evaluator)

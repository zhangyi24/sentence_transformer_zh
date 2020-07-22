"""
This example loads the pre-trained SentenceTransformer model 'bert-base-nli-mean-tokens' from the server.
It then fine-tunes this model for some epochs on the CSTS-B dataset.

Note: In this example, you must specify a SentenceTransformer model.
If you want to fine-tune a huggingface/transformers model like bert-base-uncased, see training_nli.py and training_stsbenchmark.py
"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))

import argparse
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import CSTSBenchmarkDataReader
import logging
from datetime import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default='output/bert-base-chinese-mean-cmnli', type=str,
                        help="model name or model dir")
    parser.add_argument("--num_epochs", "-e", default=10, type=int, help="number of epochs")
    args = parser.parse_args()
    model_name = args.model
    num_epochs = args.num_epochs
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout

    train_batch_size = 16
    model_save_path = 'output/%s-csts-%s' % (model_name.rstrip("/").split('/')[-1], datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    csts_reader = CSTSBenchmarkDataReader('../datasets/CSTS-B', normalize_scores=True)

    # Load a pre-trained sentence transformer model
    model = SentenceTransformer(model_name)

    # Convert the dataset to a DataLoader ready for training
    logging.info("Read CSTS-B  train dataset")
    train_data = SentencesDataset(csts_reader.get_examples('cnsd-sts-train.txt'), model)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    logging.info("Read CSTS-B dev dataset")
    dev_data = SentencesDataset(examples=csts_reader.get_examples('cnsd-sts-dev.txt'), model=model)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
    evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)

    # Configure the training. We skip evaluation in this example
    warmup_steps = math.ceil(len(train_data) * num_epochs / train_batch_size * 0.1)  # 10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=100,
              warmup_steps=warmup_steps,
              output_path=model_save_path)

    model = SentenceTransformer(model_save_path)
    test_data = SentencesDataset(examples=csts_reader.get_examples("cnsd-sts-test.txt"), model=model)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=train_batch_size)
    evaluator = EmbeddingSimilarityEvaluator(test_dataloader)
    model.evaluate(evaluator)

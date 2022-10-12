import argparse
import sys

import matplotlib.pyplot as plt
import tensorflow as tf
from datasets import SingleDataset

def get_embeddings(args: argparse.Namespace):
    model = tf.keras.models.load_model(args.model_path)
    samples = [line.rstrip() for line in open(args.test_samples, "r")]
    test_data = SingleDataset(
        sample_id_filename=args.test_samples,
        genotype_filename=args.genotypes,  # full set of genotypes
        shuffle=False,
        batch_size=args.batch_size,
        repeat=False,
    )

    for samples, gt_vectors in test_data.ds:
        embeddings = model(gt_vectors)
        for s, e in zip(samples, embeddings):
            print(s.numpy().decode("utf-8"), *e.numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        dest="model_path",
        required=True,
        help="Path to the model to evaluate",
    )
    parser.add_argument(
        "--test_samples",
        type=str,
        dest="test_samples",
        required=True,
        help="Test samples list",
    )
    parser.add_argument(
        "--genotypes", type=str, dest="genotypes", required=True, help="Genotypes file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        dest="batch_size",
        help="Batch size of inference model",
    )
    args = parser.parse_args()
    get_embeddings(args)

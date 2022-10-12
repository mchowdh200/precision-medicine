import math
import random
from itertools import combinations
from pprint import pprint
from typing import Container, Iterable, Mapping, Tuple

import numpy as np
import tensorflow as tf
from tensorflow._api.v2.data import AUTOTUNE


def nchoosek(n: int, k: int) -> int:
    return int(math.factorial(n) / (math.factorial(k) * math.factorial(n - k)))


def get_sample_names(filename: str) -> list[str]:
    """
    Loads the sample names from file
    """
    with open(filename, "r") as f:
        return f.read().splitlines()


def load_sample_ids(filename: str) -> Mapping[str, int]:
    """
    Loads the sample IDs from a file.
    :param filename: The file to load the sample IDs from.
    :return: A dict mapping sample IDs to indices.
    """
    with open(filename, "r") as f:
        sample_ids = f.read().splitlines()
    return {sample_id: i for i, sample_id in enumerate(sample_ids)}


# TODO figure out using memory mapped file to load this
def load_genotypes(filename: str) -> list[list[np.uint8]]:
    """
    load file with genotype encodings and returns a dict keyed by int id
    """
    with open(filename, "r") as f:
        genotypes = f.read().splitlines()  # string of ints with no delimiter
    return [[np.uint8(i) for i in list(g)] for g in genotypes]


# get all pairs of samples IDs along with their target distance
def get_sample_pairs(
    filename: str,
) -> list[Tuple[str, str, float]]:
    """
    Loads the sample pairs from a file.
    :param sample_ids: The sample IDs to use.
    :param filename: The file to load the sample pairs from.
    :return: A dict mapping sample pairs to target distances.
    """
    sample_pairs: list[Tuple[str, str, float]] = []

    with open(filename, "r") as f:
        next(f)  # skip header
        for line in f:
            s1, s2, d = line.split()
            sample_pairs.append((s1, s2, float(d)))
    return sample_pairs


# for each sample, get all pairs and yeild their genotype vectors and target distance
def yield_sample_pair(
    sample_ids: Mapping[str, int],
    sample_pairs: list[Tuple[str, str, float]],
    genotypes: list[list[np.uint8]],
    shuffle: bool = False,
) -> Iterable[tuple[list[np.uint8], list[np.uint8], float]]:
    """
    Yields the sample pairs and their target distances.
    :param sample_ids: sample IDs -> integer IDs.
    :param target_dict: sample pair -> distance.
    :param genotypes: integer IDs -> genotype vectors.
    :return: A generator yielding the sample pairs and their target distances.
    """
    if shuffle:
        random.shuffle(sample_pairs)
    for s1, s2, d in sample_pairs:
        yield (genotypes[sample_ids[s1]], genotypes[sample_ids[s2]], d)


def partition_samples(
    train_ratio: float,
    sample_file: str,
) -> Tuple[list[str], list[str], list[str]]:
    """
    Partitions the samples into training, validation and test sets.
    :param train_ratio: The ratio of samples to use for training.
    :param samples: The samples to partition.
    :return: A tuple of lists of sample IDs for each set.
    """
    samples = get_sample_names(sample_file)
    n = len(samples)

    # take the firsn train_ratio% of samples for training
    train_n = np.floor(train_ratio * len(samples)).astype(int)
    train_samples = samples[:train_n]

    # the rest is split 50-50 into val and test
    leftover_n = n - train_n
    val_n = leftover_n // 2
    val_samples = samples[train_n : train_n + val_n]
    test_samples = samples[train_n + val_n :]

    return train_samples, val_samples, test_samples


# TODO generator cant take advantage of multiprocessing due to race conditions
# figure this out later if the GPU doesn't get data fast enough.
# TODO add option for mapping to onehot.
class PairsDataset:
    def __init__(
        self,
        keep_samples: Container,  # leave out pairs that have samples outside this set
        sample_id_filename: str,
        sample_pair_filename: str,
        genotype_filename: str,
        shuffle: bool = False,
        encoding_type: tf.DType = tf.int8,
        target_type: tf.DType = tf.float32,
        batch_size: int = 32,
        repeat: bool = True,
    ):
        """
        TF dataset for the sample pairs and target distances.
        """
        sample_ids = load_sample_ids(sample_id_filename)
        genotypes = load_genotypes(genotype_filename)
        sample_pairs = get_sample_pairs(sample_pair_filename)

        # filter out pairs that have samples outside the keep_samples set
        sample_pairs = [
            (s1, s2, d)
            for s1, s2, d in sample_pairs
            if s1 in keep_samples and s2 in keep_samples
        ]

        self.num_pairs = len(sample_pairs) // batch_size
        self.num_variants = len(genotypes[0])

        self.ds = tf.data.Dataset.from_generator(
            lambda: yield_sample_pair(sample_ids, sample_pairs, genotypes, shuffle),
            (encoding_type, encoding_type, target_type),
            (
                tf.TensorShape([self.num_variants]),
                tf.TensorShape([self.num_variants]),
                tf.TensorShape([]),
            ),
        )
        if repeat:
            self.ds = self.ds.repeat()
        self.ds = self.ds.batch(batch_size).prefetch(10)


def load_genotypes_unsupervised(
    filename: str, keep_samples: Container[str]
) -> list[list[np.uint8]]:
    """
    Just loads a list of genotypes (encoded as int8).  No sample names.
    Each line consists of sample then genotype vector (tab separated).
    So we need to filter out the samples we don't want.
    """

    def is_keep_sample(keep_list: Container[str], sample_line: str):
        return sample_line.split()[0] in keep_list

    with open(filename, "r") as f:
        genotypes = [
            line.rstrip().split()[1:]
            for line in f.read().splitlines()
            if is_keep_sample(keep_samples, line)
        ]  # string of ints with no delimiter
    return [[np.uint8(i) for i in list(g)] for g in genotypes]


class PairsDatasetUnsupervised:
    def __init__(
        self,
        sample_id_filename: str,
        genotype_filename: str,
        shuffle: bool = False,
        encoding_type: tf.DType = tf.int8,
        batch_size: int = 32,
        repeat: bool = False,
    ):
        """
        TF dataset for the sample pairs without targets
        """
        keep_samples = get_sample_names(sample_id_filename)
        genotypes = load_genotypes_unsupervised(genotype_filename, keep_samples)

        self.num_pairs = nchoosek(len(genotypes), 2) // batch_size
        self.num_variants = len(genotypes[0])

        if shuffle:
            generator = lambda: combinations(
                random.sample(genotypes, len(genotypes)), 2
            )
        else:
            generator = lambda: combinations(genotypes, 2)

        self.ds = tf.data.Dataset.from_generator(
            generator,
            (encoding_type, encoding_type),
            (
                tf.TensorShape([self.num_variants]),
                tf.TensorShape([self.num_variants]),
            ),
        )
        if repeat:
            self.ds = self.ds.repeat()
        self.ds = self.ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


class SingleDataset:
    def __init__(
        self,
        sample_id_filename: str,
        genotype_filename: str,
        shuffle: bool = False,
        batch_size: int = 32,
        repeat: bool = True,
    ):
        """
        TF dataset for single samples.
        Each element is a tuple of (sample_id, genotype_vector)
        """
        keep_samples = get_sample_names(sample_id_filename)
        with open(genotype_filename, "r") as f:
            genotypes = [
                (line.split()[0], [float(x) for x in line.rstrip().split()[1:]])
                for line in f.read().splitlines()
                if line.split()[0] in keep_samples
            ]

        # create dataset that yeilds tuples of (sample_id, genotype_vector)
        sample_ds = tf.data.Dataset.from_tensor_slices([g[0] for g in genotypes])
        genotype_ds = tf.data.Dataset.from_tensor_slices([g[1] for g in genotypes])
        self.ds = tf.data.Dataset.zip((sample_ds, genotype_ds))

        self.num_variants = len(genotypes[0])
        # self.ds = tf.data.Dataset.from_tensor_slices(data)
        if repeat:
            self.ds = self.ds.repeat()
        if shuffle:
            self.ds = self.ds.shuffle(buffer_size=len(genotypes))
        self.ds = self.ds.batch(batch_size).prefetch(10)


if __name__ == "__main__":
    train_samples = "/home/murad/data/toy_model_data/new_model/training.samples"
    sample_id_filename = "/home/murad/data/toy_model_data/new_model/all.samples"
    genotype_filename = (
        "/home/murad/data/toy_model_data/new_model/chr8-30x.seg.86.encoded"
    )
    batch_size = 128

    train_ds = PairsDatasetUnsupervised(
        sample_id_filename=train_samples,
        genotype_filename=genotype_filename,
        shuffle=True,
        repeat=False,
        batch_size=batch_size,
    )
    print(train_ds.num_pairs)

    for i, (s1, s2) in enumerate(train_ds.ds):
        print(i, s1.shape, s2.shape)

import argparse
import sys

import tensorflow as tf
from datasets import PairsDatasetUnsupervised
from models import SiameseNN

# load the dataset
# TODO split the data into train and test at the file level
# - do the split by sample name
# - take 90% of samples for training and 5%/5% for val/test
# - use grep -v to hold out samples from the training/val/test sets

if __name__ == "__main__":

    # train_samples, val_samples, test_samples = partition_samples(
    #     train_ratio=0.9, sample_file="/home/murad/data/toy_model_data/ALL.sampleIDs"
    # )
    # # log the train/val/test samples
    # with open("train_samples.txt", "w") as f:
    #     f.write("\n".join(train_samples))
    # with open("val_samples.txt", "w") as f:
    #     f.write("\n".join(val_samples))
    # with open("test_samples.txt", "w") as f:
    #     f.write("\n".join(test_samples))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_samples",
        type=str,
        dest="train_samples",
        required=True,
        help="File containing the sample IDs to use for training.",
    )
    parser.add_argument(
        "--val_samples",
        type=str,
        dest="val_samples",
        required=True,
        help="File containing the sample IDs to use for validation.",
    )
    parser.add_argument(
        "--genotype_filename",
        type=str,
        dest="genotype_filename",
        required=True,
        help="File containing the genotypes.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        dest="batch_size",
        default=128,
        help="The batch size to use for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        dest="epochs",
        default=10,
        help="The number of epochs to train for.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        dest="learning_rate",
        default=0.001,
        help="The learning rate to use for training.",
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        dest="model_save_path",
        default="model.h5",
        help="The path to save the model to.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        dest="embedding_dim",
        default=128,
        help="The dimension of the embedding layer.",
    )
    args = parser.parse_args()

    with open(args.train_samples, "r") as f:
        train_samples = f.read().splitlines()
    with open(args.val_samples, "r") as f:
        val_samples = f.read().splitlines()

    # TODO test to make sure there is no train/test leakage
    # TODO also the filtering is slow.  Not  a big deal, but I would
    # like to be able to do it faster.
    train_data = PairsDatasetUnsupervised(
        sample_id_filename=args.train_samples,
        genotype_filename=args.genotype_filename,
        batch_size=args.batch_size,
        shuffle=True,
        repeat=True,
        encoding_type=tf.float32,
    )
    val_data = PairsDatasetUnsupervised(
        sample_id_filename=args.val_samples,
        genotype_filename=args.genotype_filename,
        batch_size=args.batch_size,
        shuffle=True,
        repeat=True,
        encoding_type=tf.float32,
    )

    print(f"{train_data.num_pairs=}")
    print(f"{val_data.num_pairs=}")

    # get base model
    # TODO just wrap this into the siamze model class?
    # base_model = resnet_model(shape=(train_data.num_variants, 1))

    # siamese_net = build_siamese_network(
    #     base_model=base_model,
    #     vector_size=train_data.num_variants,
    # )
    # training_model = SiameseModel(siamese_net)
    training_model = SiameseNN(
        input_dim=(train_data.num_variants, 1),
        embedding_dim=args.embedding_dim,
    )
    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        run_eagerly=True,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.model_save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]
    training_model.fit(
        train_data.ds,
        epochs=args.epochs,
        steps_per_epoch=train_data.num_pairs//10,
        # steps_per_epoch=1,
        validation_data=val_data.ds,
        validation_steps=val_data.num_pairs//5,
        # validation_steps=1,
        callbacks=callbacks,
    )

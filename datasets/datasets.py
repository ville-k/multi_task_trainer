# Copyright (C) 2018, Ville Kallioniemi <ville.kallioniemi@gmail.com>

from collections import namedtuple
import numpy as np
from tensorflow.contrib.keras import datasets
import tensorflow as tf

"""
Set of datasets used to train and evaluate a model.

train (tf.data.Dataset) Training set.
test (tf.data.Dataset): Test set.
labels (list of str): List of unique labels. 
"""
DatasetSet = namedtuple("DatasetSet", [
    "train",
    "test",
    "labels"
])


def keras_to_tensorflow(train, test):
    """Transform Keras numpy datasets to a a DatasetSet."""
    labels = np.unique(train[1])

    train_set = tf.data.Dataset.from_tensor_slices({
        "image": train[0],
        "label": np.squeeze(train[1])
    })
    test_set = tf.data.Dataset.from_tensor_slices({
        "image": test[0],
        "label": np.squeeze(test[1])
    })

    return DatasetSet(train_set, test_set, labels)


class DatasetSets(object):
    """Facade for loading datasets in uniform format."""
    @classmethod
    def load_cifar10(cls):
        """Return DatasetSet comprising of CIFAR-10 images and labels."""
        train, test = datasets.cifar10.load_data()
        return keras_to_tensorflow(train, test)

    @classmethod
    def load_mnist(cls):
        """Return DatasetSet comprising of MNIST images and labels."""
        train, test = datasets.mnist.load_data()
        return keras_to_tensorflow(train, test)

# Copyright (C) 2018, Ville Kallioniemi <ville.kallioniemi@gmail.com>

import numpy as np
import tensorflow as tf

from datasets import DatasetSets
from trainer import Head, Task


def create_head(head_name, out, labels, class_count):
    """Create an object classification head.

    Args:
        head_name (str): Name used to distihguish this head in the graph and metrics collections.
        out (tf.Tensor): Output from the body of the network.
        labels (tf.Tensor): Labels of type tf.uint64 compatible with this head.
        class_count (int): Number of classes to classify to.
    """
    with tf.name_scope(head_name):
        with tf.name_scope("labels_to_targets"):
            targets = tf.one_hot(labels, depth=class_count,
                                 name='one_hot_targets')

        head_collections = [head_name]
        with tf.name_scope("neck"):
            out = tf.layers.conv2d(
                out, filters=128, kernel_size=(3, 3), activation=tf.nn.relu)
            out = tf.layers.flatten(out)

        with tf.name_scope("classify"):
            out = tf.layers.dense(out, 1024, activation=tf.nn.relu)
            logits = tf.layers.dense(out, class_count)
            probabilities = tf.nn.softmax(logits)
            predictions = tf.argmax(probabilities, 1)
            tf.summary.histogram('activations', probabilities,
                                 collections=head_collections)

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=targets))

        with tf.name_scope("metrics"):
            correct = tf.equal(labels, predictions)
            accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))
            tf.summary.scalar('accuracy', accuracy,
                              collections=head_collections)
            tf.summary.scalar('loss', loss, collections=head_collections)
            metrics = {
                "loss": loss,
                "accuracy": accuracy,
            }

        summary_op = tf.summary.merge_all(key=head_name)
        optimizer = tf.train.AdamOptimizer(1e-4)

    return Head(name=head_name, optimizer=optimizer, loss=loss, metrics=metrics, summaries=summary_op)


def preprocess(example):
    """Preprocess examples to match model input layer shape and type."""
    image = tf.cast(example['image'], tf.float32)
    label = tf.cast(example['label'], tf.int64)

    return {
        "image": image,
        "label": label
    }


def create_task():
    """
    Create task that classifies objects based on the cifar10 dataset.

    Returns:
        (Task): Task with an object classifier model head.
    """
    name = "object_classifier"
    cifar = DatasetSets.load_cifar10()

    def build_head(body_output, labels): return create_head(
        name, body_output, labels, len(cifar.labels))

    return Task(
        build_head=build_head,
        training_set=cifar.train.shuffle(buffer_size=1000).map(preprocess),
        test_set=cifar.test.shuffle(buffer_size=1000).map(preprocess),
    )

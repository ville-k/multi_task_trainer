# Copyright (C) 2018, Ville Kallioniemi <ville.kallioniemi@gmail.com>
"""Interfaces used for training models."""

from collections import namedtuple

"""
Model head makes up the task specific layers of a model.

name (str): Name of the head. Use for naming collections and graph variables.
optimizer (tf.train.Optimizer): Optimizer to use to minimize the loss of this head.
loss (tf.Tensor): Loss function.
metrics (dict of str: tf.Tensor): Metric names mapped to tensors for calculating them.
summaries (tf.Tensor of tf.string): Summaries specific to this encoded as a string (from
    tf.merge_all.)
"""
Head = namedtuple("Head", field_names=[
    "name",
    "optimizer",
    "loss",
    "metrics",
    "summaries"
])


"""
Task is used to define a model head and it's datasets.

build_head (function): Function that takes body output (tf.Tensor) and target labels (tf.Tensor)
    as argument and returns a Head.
training_set (tf.data.Dataset): Dataset used for training the model_head of this task.
test_set (tf.data.Dataset): Dataset to test the performance of the model_head against.
"""
Task = namedtuple("Task", field_names=[
    "build_head",
    "training_set",
    "test_set",
])

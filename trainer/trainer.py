# Copyright (C) 2018, Ville Kallioniemi <ville.kallioniemi@gmail.com>

import os
import random

from collections import namedtuple
import tensorflow as tf


class Trainer(object):
    """Trainer capable of training multiple tasks simultaneously."""

    def __init__(self, model_constructor, training_dir, batch_size, tasks):
        """Create a trainer.

        Args:
            model_constructor (function): Function accepting a tf.Tensor as input - used for
                creating the body of the model.
            training_dir (str): Path to a directory to use for saving training checkpoints and 
                metrics.
            batch_size (int): Size of a minibatch to use during training.
            tasks (list of `Task`): List of tasks to train.
        """
        assert batch_size > 0, "Batch size should be greater than zero."
        assert len(tasks) > 0, "Should provide at least one task to train."
        self._training_dir = training_dir
        self._model_constructor = model_constructor
        self._tasks = tasks
        self._batch_size = batch_size

    def train(self, max_epochs):
        """Train tasks.

        Args:
            max_epochs (int): Number of times traiing set of each task is 
        """
        assert max_epochs > 0, "Max epochs should be greater than zero."
        with tf.Session() as session:
            with tf.name_scope("examples"):
                # Assume all datasets produce outputs of same shape and type.
                dataset = self._tasks[0].training_set.batch(self._batch_size)
                dataset_handle = tf.placeholder(tf.string, shape=[])
                iterator = tf.data.Iterator.from_string_handle(
                    dataset_handle, dataset.output_types, dataset.output_shapes
                )
                next_batch = iterator.get_next()

            with tf.variable_scope("body"):
                body_output = self._model_constructor(next_batch["image"])

            global_step = tf.Variable(0, name='global_step', trainable=False)
            with tf.variable_scope("heads"):
                trainees = self._create_trainees(
                    session, global_step, body_output, next_batch, max_epochs)

            saver = tf.train.Saver(max_to_keep=2)
            if os.path.isdir(self._training_dir) and tf.train.checkpoint_exists(self._training_dir):
                latest = tf.train.latest_checkpoint(self._training_dir)
                print(
                    "restoring training session from {} - {}".format(self._training_dir, latest))
                saver.restore(
                    session, tf.train.latest_checkpoint(self._training_dir))
            else:
                print("initializing variables")
                init_op = tf.global_variables_initializer()
                session.run(init_op)

            train_writer = tf.summary.FileWriter(os.path.join(
                self._training_dir, "train"), session.graph)
            try:
                step = self._train(saver, train_writer, session,
                                   global_step, dataset_handle, trainees)
            except KeyboardInterrupt:
                print("Interrupted, exiting.")

    def _create_trainees(self, session, global_step, body_output, next_batch, max_epochs):
        trainees = []
        for task in self._tasks:
            head = task.build_head(body_output, next_batch['label'])

            training_iterator = task.training_set.repeat(max_epochs).batch(
                self._batch_size).make_initializable_iterator()
            training_handle = session.run(training_iterator.string_handle())
            session.run(training_iterator.initializer)

            testing_iterator = task.test_set.batch(
                self._batch_size).make_initializable_iterator()
            testing_handle = session.run(testing_iterator.string_handle())
            session.run(testing_iterator.initializer)

            optimize = head.optimizer.minimize(
                head.loss, global_step=global_step)

            trainee = _Trainee(head=head, optimize=optimize,
                               training_handle=training_handle,
                               testing_handle=testing_handle)
            trainees.append(trainee)

        return trainees

    def _train(self, saver, train_writer, session, global_step, dataset_handle, trainees):
        step = None
        try:
            while True:
                try:
                    summaries = []
                    for trainee in trainees:
                        head = trainee.head
                        summary, loss, step, metrics, _ = session.run([
                            head.summaries, head.loss, global_step, head.metrics, trainee.optimize],
                            feed_dict={
                                dataset_handle: trainee.training_handle,
                        }
                        )
                        summaries.append(summary)
                        print(
                            "{:>5} train heads/{}: loss: {:2.3f},  metrics: {}".format(step, head.name, loss, metrics))

                    for summary in summaries:
                        train_writer.add_summary(summary, step)

                except tf.errors.OutOfRangeError:
                    print("End of training dataset.")
                    return step
        finally:
            if step is not None:
                saver.save(session, os.path.join(
                    self._training_dir, 'train'), global_step=step)


"""
Task head being trained by the Trainer.

Args:
    head (Head): Head of a model.
    optimize (tf.Operation): Operation for minimizing the loss of the head.
    training_handle (tf.Tensor) Tensor of type tf.string - used to identify training dataset.
    testing_handle (tf.Tensor) Tensor of type tf.string - used to identify test dataset.
"""
_Trainee = namedtuple('Trainee', field_names=[
    'head',
    'optimize',
    'training_handle',
    'testing_handle',
])

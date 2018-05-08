import os
import sys

import argparse
import importlib
import importlib.util

from tasks import classify_digits, classify_objects
from trainer import Head, Task, Trainer


def load_model_definition(path):
    """Load model architecture from a Python file dynamically."""
    spec = importlib.util.spec_from_file_location("model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Trainer for image classification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='batch size to use for training')
    parser.add_argument('--epochs', default=10, type=int,
                        help='maximum number of epochs to train for')
    # TODO: support joint training.
    parser.add_argument('--mode', default='altarnate', type=str, choices=('alternate', 'joint'),
                        help="""
'alternate' = alternate task for each batch - when datasets are different for each task.\n
'joint' = minimize sum of task losses - dataset with overlapping labels for each task.
""")
    parser.add_argument('--training_dir', default='outputs', type=str,
                        help='directory for storing training session information')
    parser.add_argument('model_file', type=str,
                        help='path to python model file to use')
    options = parser.parse_args(args=sys.argv[1:])

    print("importing model description from file {}".format(options.model_file))
    training_dir = os.path.join(
        options.training_dir,
        os.path.splitext(os.path.basename(options.model_file))[0]
    )

    model_constructor = load_model_definition(options.model_file)

    tasks = [
        classify_digits.create_task(),
        classify_objects.create_task(),
    ]

    trainer = Trainer(
        model_constructor=model_constructor,
        training_dir=training_dir,
        batch_size=options.batch_size,
        tasks=tasks
    )

    trainer.train(max_epochs=options.epochs)

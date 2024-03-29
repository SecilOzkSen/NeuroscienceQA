import os
from trapper.training.train import run_experiment
import logging
import sys

EXPERIMENT_NAME = "roberta-base-squad2"

WORKING_DIR = os.getcwd()
PROJECT_ROOT = os.path.dirname(os.path.dirname(WORKING_DIR))
EXPERIMENT_DIR = os.path.join(WORKING_DIR, EXPERIMENT_NAME)
CONFIG_PATH = os.path.join(WORKING_DIR, "experiment.jsonnet")  # default experiment params

MODEL_DIR = os.path.join(EXPERIMENT_DIR, "model")
CHECKPOINT_DIR = os.path.join(EXPERIMENT_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, "outputs")

ext_vars = {
    # Used to feed the jsonnet config file with file paths
    "OUTPUT_PATH": OUTPUT_DIR,
    "CHECKPOINT_PATH": CHECKPOINT_DIR
}

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                        level=logging.INFO, stream=sys.stdout)
    result = run_experiment(
        config_path=CONFIG_PATH,
        ext_vars=ext_vars,
    )



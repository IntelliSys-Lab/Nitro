from typing import Dict, Tuple
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from custom_test_class import Test
import logging
import ray


if __name__ == "__main__":

    ray.init(
        log_to_driver=False,
        configure_logging=True,
        logging_level=logging.ERROR
    )

    t = Test()
    t.reset_trainer()
    train_results = t.trainer.train()
    print(train_results)

    ray.shutdown()

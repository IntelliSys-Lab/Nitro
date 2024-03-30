from typing import Dict, Tuple
import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms import ppo, appo, dqn, ddpg, apex_dqn, apex_ddpg, impala
from ray.rllib.algorithms.pg.pg import PGConfig
import config
# from custom_callbacks import CustomCallbacks


class CustomCallbacks(DefaultCallbacks):
    '''
    Customized callbacks and metrics
    '''
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        pass

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )
        pass

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        if worker.config.batch_mode == "truncate_episodes":
            # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["default_policy"].batches[
                -1
            ]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called "
                "after episode is done!"
            )
        pass

    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs):
        pass

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        pass

    def on_learn_on_batch(
        self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    ) -> None:
        # grads_big, _ = policy.compute_gradients(train_batch)
        # grads_small, _ = policy.compute_gradients(train_batch)

        # result["G_big_norm"] = grads_big
        # result["G_small_norm"] = grads_small
        pass

    def on_postprocess_trajectory(
        self,
        *,
        worker: RolloutWorker,
        episode: Episode,
        agent_id: str,
        policy_id: str,
        policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[str, Tuple[Policy, SampleBatch]],
        **kwargs
    ):
        pass

class Test:

    def __init__(self):
        self.env_name = "Hopper-v3"
        self.rollout_fragment_length = 256
        self.algo_name = "ppo"

        self.trainer_config = self.init_trainer_config()
        self.trainer = None

    def init_trainer_config(self):
        # Init with max actors
        num_rollout_workers = config.num_rollout_workers_max
        train_batch_size = num_rollout_workers * self.rollout_fragment_length

        trainer_config = (
            ppo.PPOConfig()
            .framework(framework=config.framework)
            .callbacks(callbacks_class=CustomCallbacks)
            .environment(env=self.env_name)
        #     .resources(
        #         num_gpus=config.num_gpus_for_local_worker,
        #         num_cpus_for_local_worker=config.num_cpus_for_local_worker,
        #         num_cpus_per_worker=config.num_cpus_per_worker,
        #         num_gpus_per_worker=config.num_gpus_per_worker,
        #     )
        #     .rollouts(
        #         rollout_fragment_length=self.rollout_fragment_length,
        #         num_rollout_workers=num_rollout_workers,
        #         num_envs_per_worker=config.num_envs_per_worker_min,
        #         # batch_mode="complete_episodes",
        #         batch_mode="truncate_episodes",
        #     )
        #     .debugging(
        #         log_level="ERROR",
        #         logger_config={"type": ray.tune.logger.NoopLogger},
        #         log_sys_usage=False
        #     ) # Disable all loggings to save time
        #     .training(
        #         train_batch_size=train_batch_size,
        #         sgd_minibatch_size=train_batch_size,
        #         num_sgd_iter=config.num_sgd_iter,
        #         clip_param=config.clip_param,
        #         vf_clip_param=config.vf_clip_param,
        #         entropy_coeff=config.entropy_coeff,
        #         kl_coeff=config.kl_coeff,
        #     )
        #     .evaluation(
        #         evaluation_interval=config.evaluation_interval
        #     )
        )

        return trainer_config

    def reset_trainer(self):
        self.trainer = self.trainer_config.build()


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

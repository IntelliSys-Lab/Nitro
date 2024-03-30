import ray
from ray.rllib.algorithms import ppo, appo, dqn, ddpg, apex_dqn, apex_ddpg
import csv
import numpy as np
import torch
import time
import collections
import logging
import config
import utils
from env import Environment


def experiment(
    scheduler_name,
    is_serverless
):
    # Start training
    for exp_id in range(config.max_exp):
        for algo_name in config.algos:
            for env_name in config.envs.keys():
                for ckpt_round in range(1, config.boost_ckpt_round):
                    # Set up environment
                    env = Environment(
                        scheduler_name=scheduler_name,
                        algo_name=algo_name,
                        env_name=env_name,
                        target_reward=config.envs[env_name]["max_reward"],
                        budget=config.envs[env_name]["budget"],
                        stop_min_round=config.stop_min_round,
                        stop_max_round=config.stop_max_round,
                        stop_num_results=config.stop_num_results,
                        stop_cv=config.stop_cv,
                        stop_grace_period=config.stop_grace_period,
                        rollout_fragment_length=config.envs[env_name]["rollout_fragment_length"],
                        is_serverless=is_serverless,
                        is_env_discrete=config.envs[env_name]["is_env_discrete"],
                    )

                    # Start training
                    for episode_id in range(config.max_episode_eval):
                        state, mask, info = env.reset()
                        env.load("{}/{}{}".format(config.boost_ckpt_folder, config.boost_ckpt_prefix, ckpt_round))

                        csv_round = [
                            [
                                "round_id",
                                "duration",
                                "num_rollout_workers",
                                "num_envs_per_worker",
                                "episodes_this_iter",
                                "learner_time", 
                                "actor_time",
                                "eval_reward",
                                "learner_loss",
                                "cost",
                                "kl",
                                "fim",
                                "logp_ratio",
                                "entropy",
                            ]
                        ]

                        round_id = ckpt_round

                        action = {}
                        action["num_rollout_workers"] = config.num_rollout_workers_max
                        action["num_envs_per_worker"] = config.num_envs_per_worker_min

                        round_done = False
                        while round_done is False:
                            # Check if boost ends 
                            if round_id - ckpt_round >= config.boost_window_size:
                                action["num_rollout_workers"] = config.num_rollout_workers_min
                            
                            next_state, next_mask, reward, done, info = env.step(
                                round_id=round_id,
                                action=action
                            )

                            episodes_this_iter = info["episodes_this_iter"]
                            eval_reward = info["eval_reward"]
                            logp_ratio = info["logp_ratio"]
                            logp_ratio_min, logp_ratio_mean, logp_ratio_max = utils.process_logp_ratio(logp_ratio)

                            csv_round.append(
                                [
                                    round_id,
                                    info["duration"],
                                    action["num_rollout_workers"],
                                    action["num_envs_per_worker"],
                                    episodes_this_iter,
                                    info["learner_time"],
                                    info["actor_time"],
                                    eval_reward,
                                    info["learner_loss"],
                                    env.cost,
                                    info["kl"],
                                    info["fim"],
                                    logp_ratio_mean,
                                    info['entropy'],
                                ]
                            )

                            print("")
                            print("******************")
                            print("******************")
                            print("******************")
                            print("")
                            print("Running {}, algo {}, env {}, episode {}".format(scheduler_name, algo_name, env_name, episode_id))
                            print("round_id: {}".format(info["round_id"]))
                            print("duration: {}".format(info["duration"]))
                            print("action: {}".format(action))
                            print("episodes_this_iter: {}".format(episodes_this_iter))
                            print("eval_reward: {}".format(info["eval_reward"]))
                            print("cost: {}".format(env.cost))

                            if done:
                                utils.export_csv(
                                    exp_id=exp_id,
                                    episode_id=episode_id,
                                    scheduler_name=scheduler_name,
                                    env_name=env_name, 
                                    algo_name=algo_name, 
                                    csv_name=ckpt_round,
                                    csv_file=csv_round
                                )

                                round_done = True
                            
                            state = next_state
                            mask = next_mask
                            round_id = round_id + 1


if __name__ == "__main__":
    scheduler_name = "boost_eval"
    is_serverless = False
    # is_serverless = True

    print("")
    print("**********")
    print("**********")
    print("**********")
    print("")
    ray.init(
        log_to_driver=False,
        configure_logging=True,
        logging_level=logging.ERROR
    )

    print("Start training...")
    print("Running {}, is serverless? {}".format(scheduler_name, is_serverless))
    experiment(
        scheduler_name=scheduler_name,
        is_serverless=is_serverless
    )

    ray.shutdown()
    print("")
    print("{} training finished!".format(scheduler_name))
    print("")
    print("**********")
    print("**********")
    print("**********")
import ray
from ray.rllib.algorithms import ppo, appo, dqn, ddpg, apex_dqn, apex_ddpg
import csv
import numpy as np
import torch
import time
import logging
import config
import utils
from minions_agent import PPOAgent
from env import Environment


def experiment(
    scheduler_name,
    is_serverless
):
    # Start training
    for exp_id in range(config.max_exp):
        for algo_name in config.algos:
            for env_name in config.envs.keys():
                for actor_ratio in config.actor_ratio:
                    for recover_round in config.recover_round:
                        # Set up environment
                        env = Environment(
                            scheduler_name=scheduler_name,
                            algo_name=algo_name,
                            env_name=env_name,
                            target_reward=config.envs[env_name]["max_reward"],
                            budget=config.envs[env_name]["budget"],
                            min_round=recover_round,
                            max_round=config.max_round,
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
                            # Critical period parameters
                            action = int(config.num_max_rollout_workers * actor_ratio)

                            csv_round = [
                                [
                                    "round_id",
                                    "duration",
                                    "num_actor",
                                    "learner_time", 
                                    "actor_time",
                                    "eval_reward",
                                    "learner_loss",
                                    "cost",
                                    "kl",
                                    "fim"
                                ]
                            ]
                            
                            round_done = False
                            round_count = 0
                            while round_done is False:
                                # Check if recover round
                                if round_count >= recover_round:
                                    action = config.num_max_rollout_workers
                                next_state, next_mask, reward, done, info = env.step(
                                    action=action
                                )

                                csv_round.append(
                                    [
                                        info["round_id"],
                                        info["duration"],
                                        action,
                                        info["learner_time"],
                                        info["actor_time"],
                                        info["eval_reward"],
                                        info["learner_loss"],
                                        env.cost,
                                        info["kl"],
                                        info["fim"]
                                    ]
                                )

                                print("")
                                print("Running {}, algo {}, env {}, episode {}".format(scheduler_name, algo_name, env_name, episode_id))
                                print("round_id: {}".format(info["round_id"]))
                                print("duration: {}".format(info["duration"]))
                                print("action: {}".format(action))
                                print("reward: {}".format(reward))
                                print("learner_time: {}".format(info["learner_time"]))
                                print("actor_time: {}".format(info["actor_time"]))
                                print("eval_reward: {}".format(info["eval_reward"]))
                                print("eval_reward_cv: {}".format(info["eval_reward_cv"]))
                                print("learner_loss: {}".format(info["learner_loss"]))
                                print("cost: {}".format(env.cost))
                                print("kl: {}".format(info["kl"]))
                                print("fim: {}".format(info["fim"]))
                                print("budget remain: {}".format(env.budget - env.cost))

                                if done:
                                    utils.export_csv(
                                        exp_id=exp_id,
                                        episode_id=episode_id,
                                        scheduler_name=scheduler_name,
                                        env_name=env_name, 
                                        algo_name=algo_name, 
                                        csv_name="motivate_{}_{}".format(actor_ratio, recover_round),
                                        csv_file=csv_round
                                    )

                                    round_done = True
                                    
                                state = next_state
                                mask = next_mask
                                round_count = round_count + 1


if __name__ == "__main__":
    scheduler_name = "motivate"
    # is_serverless = False
    is_serverless = True

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
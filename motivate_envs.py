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
                for (num_actors, num_envs) in zip(config.actors_list, config.envs_list):
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

                    prev_j_list = []
                    j_k_sum = None

                    # Start training
                    for episode_id in range(config.max_episode_eval):
                        state, mask, info = env.reset()
                        action = {
                            "num_rollout_workers": num_actors,
                            "num_envs_per_worker": num_envs
                        }

                        csv_round = [
                            [
                                "round_id",
                                "duration",
                                "num_actors",
                                "num_envs",
                                "episodes_this_iter",
                                "learner_time", 
                                "actor_time",
                                "eval_reward",
                                "learner_loss",
                                "cost",
                                "kl",
                                "fim",
                                "logp_ratio",
                            ]
                        ]

                        round_id = 1

                        round_done = False
                        while round_done is False:
                            # Compute pac m and the action
                            m = utils.pac_m(
                                delta=config.delta, 
                                epsilon=config.epsilon, 
                                prev_j_list=prev_j_list,
                                j_k_sum=j_k_sum,
                                beta=config.beta,
                                alpha_scaling=config.envs[env_name]["alpha_scaling"],
                            )

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
                                    info["round_id"],
                                    info["duration"],
                                    action["num_rollout_workers"],
                                    action["num_envs_per_worker"],
                                    episodes_this_iter,
                                    info["learner_time"],
                                    info["actor_time"],
                                    info["eval_reward"],
                                    info["learner_loss"],
                                    env.cost,
                                    info["kl"],
                                    info["fim"],
                                    logp_ratio_mean,
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
                            print("eval_reward_cv: {}".format(info["eval_reward_cv"]))
                            print("cost: {}".format(env.cost))
                            print("---")
                            print("PAC related:")
                            print("m: {}".format(m))
                            try:
                                print("m_action: {}".format(int(m/config.envs[env_name]["rollout_fragment_length"])+1))
                            except:
                                print("m_action not applicable!")

                            if done:
                                utils.export_csv(
                                    exp_id=exp_id,
                                    episode_id=episode_id,
                                    scheduler_name=scheduler_name,
                                    env_name=env_name, 
                                    algo_name=algo_name, 
                                    csv_name="{}x{}".format(num_actors, num_envs),
                                    csv_file=csv_round
                                )

                                round_done = True
                            
                            prev_j_list = info["episode_reward"]
                            j_k_sum = info["j_k_sum"]
                            # print("prev_j_list: {}".format(prev_j_list))
                            # print("logp_ratio: {}".format(logp_ratio))
                                
                            state = next_state
                            mask = next_mask
                            round_id = round_id + 1


if __name__ == "__main__":
    scheduler_name = "motivate_envs"
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
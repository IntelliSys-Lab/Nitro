import numpy as np
import torch
import time
import csv
import collections
import logging
import config
import utils
import ray
from env import Environment


def experiment(
    scheduler_name,
    is_serverless
):
    # Start training
    for exp_id in range(config.max_exp):
        for algo_name in config.algos:
            for env_name in config.envs.keys():
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
                            "eff",
                            "kl",
                            "gns",
                            "gns_score",
                        ]
                    ]

                    round_id = 1

                    gns_ewma = {}
                    gns_ewma["S_biased"] = None
                    gns_ewma["G_biased"] = None

                    action = {}
                    action["num_rollout_workers"] = config.num_rollout_workers_min
                    action["num_envs_per_worker"] = config.num_envs_per_worker_min

                    gns_history = []

                    old_stats = {}
                    old_stats["eval_reward"] = None
                    old_stats["cost"] = None

                    round_done = False
                    while round_done is False:
                        next_state, next_mask, reward, done, info = env.step(
                            round_id=round_id,
                            action=action
                        )

                        #
                        # Update current stats
                        #

                        # Update gns via EWMA
                        for k in gns_ewma.keys():
                            if gns_ewma[k] is None:
                                gns_ewma[k] = info[k]
                            else:
                                gns_ewma[k] = info[k] * config.gns_ewma_alpha + gns_ewma[k] * (1 - config.gns_ewma_alpha)

                        gns = gns_ewma["S_biased"] / gns_ewma["G_biased"]
                        gns_history.append(gns)

                        # Compute z score
                        gns_score = utils.z_score(utils.remove_outliers(gns_history), gns)

                        # Compute effciency and update old reward/cost
                        if old_stats["eval_reward"] is None or old_stats["cost"] is None:
                            eff = 0
                        else:
                            eff = np.clip(
                                (info["eval_reward"] - old_stats["eval_reward"]) / (info["cost"] - old_stats["cost"]),
                                a_min=0,
                                a_max=None
                            )
                        old_stats["eval_reward"] = info["eval_reward"]
                        old_stats["cost"] = info["cost"]

                        #
                        # Boost if the score is beyond threshold
                        #

                        if gns_score >= config.gns_boost_score:
                            action["num_rollout_workers"] = config.num_rollout_workers_max
                        else:
                            action["num_rollout_workers"] = config.num_rollout_workers_min

                        csv_round.append(
                            [
                                round_id,
                                info["duration"],
                                action["num_rollout_workers"],
                                action["num_envs_per_worker"],
                                info["episodes_this_iter"],
                                info["learner_time"],
                                info["actor_time"],
                                info["eval_reward"],
                                info["learner_loss"],
                                info["cost"],
                                eff,
                                info["kl"],
                                gns,
                                gns_score,
                            ]
                        )

                        print("")
                        print("******************")
                        print("******************")
                        print("******************")
                        print("")
                        print("Running {}, algo {}, env {}, episode {}, exp {}".format(scheduler_name, algo_name, env_name, episode_id, exp_id))
                        print("round_id: {}".format(info["round_id"]))
                        print("duration: {}".format(info["duration"]))
                        print("action: {}".format(action))
                        print("episodes_this_iter: {}".format(info["episodes_this_iter"]))
                        print("eval_reward: {}".format(info["eval_reward"]))
                        print("gns: {}".format(gns))
                        print("gns_score: {}".format(gns_score))
                        print("cost: {}".format(info["cost"]))
                        print("eff: {}".format(eff))

                        if done:
                            utils.export_csv(
                                exp_id=exp_id,
                                episode_id=episode_id,
                                scheduler_name=scheduler_name,
                                env_name=env_name, 
                                algo_name=algo_name, 
                                csv_name="",
                                csv_file=csv_round
                            )

                            round_done = True
                        
                        state = next_state
                        mask = next_mask
                        round_id = round_id + 1


if __name__ == "__main__":
    scheduler_name = "boost_gns"
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
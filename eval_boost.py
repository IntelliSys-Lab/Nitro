import numpy as np
import copy
import torch
import time
import csv
import collections
import glob
import logging
import ray
from env import Environment
import config
import utils


def eval_boost(
    scheduler_name,
    is_serverless,
    algo_name,
    env_name,
    ckpt_path,
    json_path,
):
    # Set up environment and load checkpoint
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
        is_serverless=is_serverless,
    )

    # Start training
    state, mask, info = env.reset()
    env.load(ckpt_path)
    boost_round_id = utils.json_load(json_path)['round_id']

    csv_round = [
        [
            "round_id",
            "duration",
            "num_rollout_workers",
            "num_envs_per_worker",
            "learner_time", 
            "actor_time",
            "eval_reward_max",
            "eval_reward_mean",
            "eval_reward_min",
        ]
    ]

    round_id = boost_round_id

    action = {}
    action["num_envs_per_worker"] = config.num_envs_per_worker_min

    round_done = False
    while round_done is False:
        # Boost at this round
        if round_id == boost_round_id:
            action["num_rollout_workers"] = config.num_rollout_workers_max
        else:
            action["num_rollout_workers"] = config.num_rollout_workers_min
        
        next_state, next_mask, reward, done, info = env.step(
            round_id=round_id,
            action=action
        )

        csv_round.append(
            [
                round_id,
                info["duration"],
                action["num_rollout_workers"],
                action["num_envs_per_worker"],
                info["learner_time"],
                info["actor_time"],
                info["eval_reward_max"],
                info["eval_reward_mean"],
                info["eval_reward_min"],
            ]
        )

        print("")
        print("******************")
        print("******************")
        print("******************")
        print("")
        print("Running {}, algo {}, env {}".format(scheduler_name, algo_name, env_name))
        print("round_id: {}".format(info["round_id"]))
        print("duration: {}".format(info["duration"]))
        print("action: {}".format(action))
        print("eval_reward_mean: {}".format(info["eval_reward_mean"]))

        # # Evaluate reward surface of the updated model (just once)
        # if round_id == boost_round_id:
        #     csv_reward_surfaces = utils.eval_perturbation(
        #         round_id=round_id,
        #         env=env,
        #         grid_size=grid_size,
        #         estimate_batch=info['estimate_batch'],
        #     )

        #     # Log out as CSV
        #     utils.export_csv(
        #         scheduler_name=scheduler_name,
        #         env_name=env_name, 
        #         algo_name=algo_name, 
        #         csv_name="surface",
        #         csv_file=csv_reward_surfaces
        #     )
        
        if done:
            utils.export_csv(
                scheduler_name=scheduler_name,
                env_name=env_name, 
                algo_name=algo_name, 
                csv_name=boost_round_id,
                csv_file=csv_round
            )

            env.stop_trainer()
            round_done = True

        state = next_state
        mask = next_mask
        round_id = round_id + 1

    
if __name__ == '__main__':
    ckpt_scheduler_name = "serverful_baseline"
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

    for algo_name in config.serverful_algos:
        for env_name in config.envs.keys():
            for ckpt_filename in glob.glob("{}/{}~{}~{}~*/".format(config.ckpt_path, ckpt_scheduler_name, env_name, algo_name)):
                eval_round_id = ckpt_filename.split('~')[-1].replace('/', '')
                scheduler_name = "eval_boost_{}".format(eval_round_id)
                ckpt_path = "{}/{}~{}~{}~{}".format(config.ckpt_path, ckpt_scheduler_name, env_name, algo_name, eval_round_id)
                json_path = "{}/{}~{}~{}~{}.json".format(config.ckpt_path, ckpt_scheduler_name, env_name, algo_name, eval_round_id)
                eval_boost(
                    scheduler_name=scheduler_name,
                    is_serverless=is_serverless,
                    algo_name=algo_name,
                    env_name=env_name,
                    ckpt_path=ckpt_path,
                    json_path=json_path,
                )
            
    ray.shutdown()
    print("")
    print("**********")
    print("**********")
    print("**********")
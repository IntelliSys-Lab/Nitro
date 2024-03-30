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


def eval_reward_surface(
    scheduler_name,
    is_serverless,
    algo_name,
    env_name,
    ckpt_path,
    pickle_path,
    json_path,
    grid_size,
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
        rollout_fragment_length=config.envs[env_name]["rollout_fragment_length"],
        is_serverless=is_serverless,
        is_env_discrete=config.envs[env_name]["is_env_discrete"],
    )

    # Load checkpoints
    state, mask, info = env.reset()
    env.load(ckpt_path)
    estimate_batch = utils.pickle_load(pickle_path)
    round_id = utils.json_load(json_path)['round_id']

    # Evaluate perturbations
    csv_reward_surfaces = utils.eval_perturbation(
        round_id=round_id,
        env=env,
        grid_size=grid_size,
        estimate_batch=estimate_batch,
    )

    # Log out as CSV
    utils.export_csv(
        scheduler_name=scheduler_name,
        env_name=env_name, 
        algo_name=algo_name, 
        csv_name="surface",
        csv_file=csv_reward_surfaces
    )
        
    env.stop_trainer()

    
if __name__ == '__main__':
    ckpt_scheduler_name = "eval_trajectory"
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

    for algo_name in config.algos:
        for env_name in config.envs.keys():
            # Min and Max convexity
            for eval_type in config.eval_types:
                scheduler_name = "eval_reward_surface_{}".format(eval_type)
                ckpt_path = "{}/{}~{}~{}~{}".format(config.ckpt_path, ckpt_scheduler_name, env_name, algo_name, eval_type)
                pickle_path = "{}/{}~{}~{}~{}.pkl".format(config.ckpt_path, ckpt_scheduler_name, env_name, algo_name, eval_type)
                json_path = "{}/{}~{}~{}~{}.json".format(config.ckpt_path, ckpt_scheduler_name, env_name, algo_name, eval_type)
                eval_reward_surface(
                    scheduler_name=scheduler_name,
                    is_serverless=is_serverless,
                    algo_name=algo_name,
                    env_name=env_name,
                    ckpt_path=ckpt_path,
                    pickle_path=pickle_path,
                    json_path=json_path,
                    grid_size=config.grid_size,
                )
            
            # Fixed checkpoints
            for round_id in config.fixed_rounds:
                scheduler_name = "eval_reward_surface_{}".format(round_id)
                ckpt_path = "{}/{}~{}~{}~{}".format(config.ckpt_path, ckpt_scheduler_name, env_name, algo_name, round_id)
                pickle_path = "{}/{}~{}~{}~{}.pkl".format(config.ckpt_path, ckpt_scheduler_name, env_name, algo_name, round_id)
                json_path = "{}/{}~{}~{}~{}.json".format(config.ckpt_path, ckpt_scheduler_name, env_name, algo_name, round_id)
                eval_reward_surface(
                    scheduler_name=scheduler_name,
                    is_serverless=is_serverless,
                    algo_name=algo_name,
                    env_name=env_name,
                    ckpt_path=ckpt_path,
                    pickle_path=pickle_path,
                    json_path=json_path,
                    grid_size=config.grid_size,
                )

    ray.shutdown()
    print("")
    print("**********")
    print("**********")
    print("**********")
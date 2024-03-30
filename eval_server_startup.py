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


def eval_server_startup(
    scheduler_name,
    is_serverless,
    algo_name,
    env_name,
    pause_learner,
    eval_surface,
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

    csv_round = [
        [
            "round_id",
            "duration",
            "num_rollout_workers",
            "num_envs_per_worker",
            "episodes_this_iter",
            "learner_time", 
            "actor_time",
            "eval_reward_max",
            "eval_reward_mean",
            "eval_reward_min",
            "learner_loss",
            "cost",
            "hessian_eigen_cv",
            "hessian_eigen_ratio",
        ]
    ]

    action = {}
    action["num_rollout_workers"] = config.num_rollout_workers_max
    action["num_envs_per_worker"] = config.num_envs_per_worker_min

    hessian_history = {}
    hessian_history['cv'] = []
    hessian_history['ratio'] = []

    # Startup overhead
    if pause_learner:
        env.pause_learner()
        time.sleep(config.server_startup_time)
        env.resume_learner()
    else:
        time.sleep(config.server_startup_time)
    
    if eval_surface:
        # Evaluate perturbations after startup
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
            csv_name='surface',
            csv_file=csv_reward_surfaces
        )

    round_done = False
    while round_done is False:
        next_state, next_mask, reward, done, info = env.step(
            round_id=round_id,
            action=action
        )

        # Evaluate Hessian eigenvalues
        hessian_eigen_cv, hessian_eigen_ratio = utils.eval_hessian(
            env=env,
            estimate_batch=info['estimate_batch'],
        )
        
        csv_round.append(
            [
                round_id,
                info["duration"],
                action["num_rollout_workers"],
                action["num_envs_per_worker"],
                info["episodes_this_iter"],
                info["learner_time"],
                info["actor_time"],
                info["eval_reward_max"],
                info["eval_reward_mean"],
                info["eval_reward_min"],
                info["learner_loss"],
                info["cost"],
                hessian_eigen_cv,
                hessian_eigen_ratio,
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
        print("episodes_this_iter: {}".format(info["episodes_this_iter"]))
        print("eval_reward_mean: {}".format(info["eval_reward_mean"]))
        print("hessian_eigen_cv: {}".format(hessian_eigen_cv))
        print("hessian_eigen_ratio: {}".format(hessian_eigen_ratio))
        print("cost: {}".format(info["cost"]))

        hessian_history['cv'].append(hessian_eigen_cv)
        hessian_history['ratio'].append(hessian_eigen_ratio)

        if done:
            utils.export_csv(
                scheduler_name=scheduler_name,
                env_name=env_name, 
                algo_name=algo_name, 
                csv_name="traj",
                csv_file=csv_round
            )

            env.stop_trainer()
            round_done = True

        state = next_state
        mask = next_mask
        round_id = round_id + 1

    
if __name__ == '__main__':
    ckpt_scheduler_name = "eval_trajectory"
    scheduler_name = "eval_server_startup"
    is_serverless = False
    pause_learner = True
    eval_surface = False

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
            # Server
            ckpt_path = "{}/{}~{}~{}~10".format(config.ckpt_path, ckpt_scheduler_name, env_name, algo_name)
            pickle_path = "{}/{}~{}~{}~10.pkl".format(config.ckpt_path, ckpt_scheduler_name, env_name, algo_name)
            json_path = "{}/{}~{}~{}~10.json".format(config.ckpt_path, ckpt_scheduler_name, env_name, algo_name)
            eval_server_startup(
                scheduler_name=scheduler_name,
                is_serverless=is_serverless,
                env_name=env_name,
                algo_name=algo_name,
                pause_learner=pause_learner,
                eval_surface=eval_surface,
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
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
import numpy as np


def eval_convex_ratio(
    scheduler_name,
    is_serverless,
    algo_name,
    env_name,
    ckpt_path,
    json_path,
    num_rollout_workers,
    num_envs_per_worker,
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

    round_id = boost_round_id

    action = {}
    action["num_envs_per_worker"] = num_envs_per_worker
    action["num_rollout_workers"] = num_rollout_workers

    next_state, next_mask, reward, done, info = env.step(
        round_id=round_id,
        action=action
    )

    # Evaluate Hessian eigenvalues
    hessian_eigen_cv, hessian_eigen_ratio = utils.eval_hessian(
        env=env,
        estimate_batch=info['estimate_batch'],
    )

    env.trainer.stop()

    return {
        "hessian_eigen_ratio": hessian_eigen_ratio,
        "episode_reward": info["episode_reward"],
    }

    
if __name__ == '__main__':
    ckpt_scheduler_name = "serverful_baseline"
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

    for algo_name in config.serverful_algos:
        for env_name in config.envs.keys():
            # for ckpt_filename in glob.glob("{}/{}~{}~{}~*/".format(config.ckpt_path, ckpt_scheduler_name, env_name, algo_name)):
            #     eval_round_id = ckpt_filename.split('~')[-1].replace('/', '')
            for eval_round_id in config.envs[env_name]["eval_convex_ratio"]:
                scheduler_name = "eval_convex_ratio"
                ckpt_path = "{}/{}~{}~{}~{}".format(config.ckpt_path, ckpt_scheduler_name, env_name, algo_name, eval_round_id)
                json_path = "{}/{}~{}~{}~{}.json".format(config.ckpt_path, ckpt_scheduler_name, env_name, algo_name, eval_round_id)

                csv_convex_ratio = [
                    [
                        "num_envs_per_worker",
                        "hessian_eigen_ratio",
                        "reward",
                    ]
                ]

                for [num_rollout_workers, num_envs_per_worker] in config.Nitro_boost_candidates:
                    for eval_time in range(config.Nitro_boost_eval_time):
                        result = eval_convex_ratio(
                            scheduler_name=scheduler_name,
                            is_serverless=is_serverless,
                            algo_name=algo_name,
                            env_name=env_name,
                            ckpt_path=ckpt_path,
                            json_path=json_path,
                            num_rollout_workers=num_rollout_workers,
                            num_envs_per_worker=num_envs_per_worker,
                        )

                        for reward in result["episode_reward"]:
                            csv_convex_ratio.append(
                                [
                                    int(num_rollout_workers / config.num_rollout_workers_serverful * num_envs_per_worker),
                                    result["hessian_eigen_ratio"],
                                    reward,
                                ]
                            )

                    print("")
                    print("******************")
                    print("******************")
                    print("******************")
                    print("")
                    print("Running {}, algo {}, env {}".format(scheduler_name, algo_name, env_name))
                    print("round_id: {}".format(eval_round_id))
                    print("eval_reward_mean: {}".format(np.mean(result["episode_reward"])))

                utils.export_csv(
                    scheduler_name=scheduler_name,
                    env_name=env_name, 
                    algo_name=algo_name, 
                    csv_name=eval_round_id,
                    csv_file=csv_convex_ratio
                )
            
    ray.shutdown()
    print("")
    print("**********")
    print("**********")
    print("**********")
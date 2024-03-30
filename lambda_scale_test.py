import numpy as np
import collections
import logging
import ray
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from env import Environment
import config
import utils


def lambda_scale_test(
    scheduler_name,
    num_rollout_workers,
    algo_name,
    env_name,
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
        is_serverless=True,
    )

    # Start training
    state, mask, info = env.reset()

    payload = {
        "redis_host": config.redis_host,
        "redis_port": config.redis_port,
        "redis_password": config.redis_password,
        "algo_name": algo_name,
        "env_name": env_name,
        "num_envs_per_worker": 1,
        "rollout_fragment_length": config.envs[env_name]['rollout_fragment_length'],
    }

    invoke_overhead, query_overhead = env.scale_test(
        num_rollout_workers=num_rollout_workers,
        payload=payload,
    )

    print("")
    print("******************")
    print("******************")
    print("******************")
    print("")
    print("Running {}, algo {}, env {}".format(scheduler_name, algo_name, env_name))
    print("invoke_overhead: {}".format(invoke_overhead))
    print("query_overhead: {}".format(query_overhead))

    env.stop_trainer()

    return invoke_overhead, query_overhead

    
if __name__ == '__main__':
    scheduler_name = "lambda_scale_test"
    num_rollout_workers_list = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100, 1]
    csv_invoke_overhead = [
        [
            "Hopper",
            "Humanoid",
            "Walker2d",
            "Gravitar",
            "SpaceInvaders",
            "Qbert",
        ]
    ]
    csv_query_overhead = [
        [
            "Hopper",
            "Humanoid",
            "Walker2d",
            "Gravitar",
            "SpaceInvaders",
            "Qbert",
        ]
    ]

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

    for num_rollout_workers in num_rollout_workers_list:
        for algo_name in config.algos:
            csv_env_invoke_overhead = []
            csv_env_query_overhead = []

            for env_name in config.envs.keys():
                invoke_overhead, query_overhead = lambda_scale_test(
                    scheduler_name=scheduler_name,
                    num_rollout_workers=num_rollout_workers,
                    algo_name=algo_name,
                    env_name=env_name,
                )
                csv_env_invoke_overhead.append(invoke_overhead)
                csv_env_query_overhead.append(query_overhead)

            csv_invoke_overhead.append(csv_env_invoke_overhead)
            csv_query_overhead.append(csv_env_query_overhead)

    utils.export_csv(
        scheduler_name=scheduler_name,
        env_name="", 
        algo_name="", 
        csv_name="invoke_overhead",
        csv_file=csv_invoke_overhead
    )
    utils.export_csv(
        scheduler_name=scheduler_name,
        env_name="", 
        algo_name="", 
        csv_name="query_overhead",
        csv_file=csv_query_overhead
    )


    ray.shutdown()
    print("")
    print("**********")
    print("**********")
    print("**********")

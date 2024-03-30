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

                    # Logging per baseline
                    csv_round = {}
                    # baselines = config.verify_baselines + ["m"]
                    baselines = config.verify_baselines
                    for baseline in baselines:
                        csv_round[baseline] = [
                            [
                                "round_id",
                                "num_actor",
                                "episodes_this_iter",
                                # "eval_reward_min",
                                "eval_reward_mean",
                                # "eval_reward_max",
                                # "policy_update_min",
                                "policy_update_mean",
                                # "policy_update_max",
                                # "kl_min",
                                "kl_mean",
                                # "kl_max",
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

                        if m is None:
                            action_m = config.num_max_rollout_workers 
                        else:
                            action_m = np.clip(
                                int(m/config.envs[env_name]["rollout_fragment_length"]) + 1,
                                config.num_min_rollout_workers,
                                config.num_max_rollout_workers
                            )

                        # Freeze current policy
                        # froze_algo_state = env.trainer.__getstate__()
                        froze_policy_state = env.trainer.get_policy().get_state()

                        #
                        # Fake update
                        #

                        # if len(prev_j_list) > 0 and j_k_sum is not None and j_k_sum != 0:
                        if True:
                            for baseline in baselines:
                                episodes_this_iter_list = []
                                eval_rewards = []
                                policy_updates = []
                                kls = []

                                if baseline == "m":
                                    action = action_m
                                else:
                                    action = baseline

                                for loop_id in range(config.verify_loop_time):
                                    # Retrieve the froze policy
                                    state, mask, info = env.reset()
                                    env.trainer.get_policy().set_state(froze_policy_state)

                                    # Train one round
                                    next_state, next_mask, reward, done, info = env.step(
                                        round_id=round_id,
                                        action=action,
                                    )

                                    # Results from current round
                                    episodes_this_iter = info['episodes_this_iter']
                                    eval_reward = info["eval_reward"]
                                    logp_ratio = info["logp_ratio"]
                                    kl = info["kl"]

                                    # Process resutls
                                    if action == baseline and action < action_m:
                                        # policy_update = torch.mean(torch.clamp(
                                        #     torch.abs(1 - logp_ratio),
                                        #     min=0,
                                        #     max=env.policy_update_bound
                                        # )).numpy()
                                        policy_update = torch.mean(
                                            torch.abs(1 - logp_ratio)
                                        ).numpy()
                                    else:
                                        policy_update = torch.mean(
                                            torch.abs(1 - logp_ratio)
                                        ).numpy()
                                    
                                    # logp_ratio_min, logp_ratio_mean, logp_ratio_max = utils.process_logp_ratio(logp_ratio)
                                    # policy_update = logp_ratio_mean

                                    episodes_this_iter_list.append(episodes_this_iter)
                                    eval_rewards.append(eval_reward)
                                    policy_updates.append(policy_update)
                                    kls.append(kl)
                                    
                                    print("")
                                    print("******************")
                                    print("******************")
                                    print("******************")
                                    print("")
                                    print("Running {}, algo {}, env {}, episode {}, round_id {}, fake update loop {}".format(scheduler_name, algo_name, env_name, episode_id, round_id, loop_id))
                                    print("action_m: {}".format(action_m))
                                    print("action: {}".format(action))
                                    print("episodes_this_iter: {}".format(episodes_this_iter))
                                    print("eval_reward: {}".format(eval_reward))
                                    print("policy_update: {}".format(policy_update))
                                    
                                # Log all stats
                                csv_round[baseline].append(
                                    [
                                        round_id,
                                        action,
                                        np.mean(episodes_this_iter_list),
                                        # np.min(eval_rewards),
                                        np.mean(eval_rewards),
                                        # np.max(eval_rewards),
                                        # np.min(policy_updates),
                                        np.mean(policy_updates),
                                        # np.max(policy_updates),
                                        # np.min(kls),
                                        np.mean(kls),
                                        # np.max(kls),
                                    ]
                                )

                        #
                        # True update
                        #

                        action = action_m

                        # Retrieve the froze policy
                        # env.trainer.__setstate__(froze_algo_state)
                        env.trainer.get_policy().set_state(froze_policy_state)

                        # Train one round
                        next_state, next_mask, reward, done, info = env.step(
                            round_id=round_id,
                            action=action,
                        )

                        eval_reward = info["eval_reward"]
                        logp_ratio = info["logp_ratio"]
                        kl = info["kl"]
                        
                        print("")
                        print("******************")
                        print("******************")
                        print("******************")
                        print("")
                        print("Running {}, algo {}, env {}, episode {}, round_id {}, true update".format(scheduler_name, algo_name, env_name, episode_id, round_id))
                        print("action_m: {}".format(action_m))
                        print("action: {}".format(action))
                        print("eval_reward: {}".format(eval_reward))

                        prev_j_list = info["episode_reward"]
                        j_k_sum = info["j_k_sum"]
                        # print("prev_j_list: {}".format(prev_j_list))
                        # print("logp_ratio: {}".format(logp_ratio))
                                
                        state = next_state
                        mask = next_mask
                        round_id = round_id + 1

                        if done:
                            for baseline in csv_round.keys():
                                utils.export_csv(
                                    exp_id=exp_id,
                                    episode_id=episode_id,
                                    scheduler_name=scheduler_name,
                                    env_name=env_name, 
                                    algo_name=algo_name, 
                                    csv_name="{}".format(baseline),
                                    csv_file=csv_round[baseline]
                                )

                            round_done = True


if __name__ == "__main__":
    scheduler_name = "pac_verify"
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
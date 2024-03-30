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
                # Set up ppo agent
                agent = PPOAgent(
                    state_dim=config.state_dim,
                    action_dim=config.action_dim,
                    hidden_dims=config.hidden_dims,
                    learning_rate=config.learning_rate,
                    discount_factor=config.discount_factor,
                    ppo_clip=config.ppo_clip,
                    ppo_epoch=config.ppo_epoch,
                    value_loss_coef=config.value_loss_coef,
                    entropy_coef=config.entropy_coef
                )
                agent.load(config.model_save_path + "max_cumulative_reward_{}_{}.pth".format(env_name, algo_name))

                # Set up environment
                env = Environment(
                    scheduler_name=scheduler_name,
                    algo_name=algo_name,
                    env_name=env_name,
                    target_reward=config.envs[env_name]["max_reward"],
                    budget=config.envs[env_name]["budget"],
                    is_serverless=is_serverless
                )

                # Start training
                for episode_id in range(config.max_episode_eval):
                    state, mask, info = env.reset()
                    cumulative_reward = 0
                    total_duration = 0

                    action = config.num_max_rollout_workers

                    csv_round = [
                        [
                            "round",
                            "duration",
                            "num_actor",
                            "learner_time", 
                            "actor_time",
                            "eval_reward",
                            "learner_loss",
                            "cost"
                        ]
                    ]

                    round_done = False
                    while round_done is False:
                        action, _, value_pred, log_prob = agent.choose_action(
                            state=state, 
                            mask=mask
                        )
                        action_item = int(action.item() + 1)
                        next_state, next_mask, reward, done, info = env.step(
                            action=action_item
                        )

                        csv_round.append(
                            [
                                info["round"],
                                info["duration"],
                                action_item,
                                info["learner_time"],
                                info["actor_time"],
                                info["eval_reward"],
                                info["learner_loss"],
                                env.cost
                            ]
                        )
                        print("")
                        print("Running {}, algo {}, env {}, episode {}".format(scheduler_name, algo_name, env_name, episode_id))
                        print("round: {}".format(info["round"]))
                        print("duration: {}".format(info["duration"]))
                        print("action: {}".format(action_item))
                        print("reward: {}".format(reward))
                        print("learner_time: {}".format(info["learner_time"]))
                        print("actor_time: {}".format(info["actor_time"]))
                        print("eval_reward: {}".format(info["eval_reward"]))
                        print("learner_loss: {}".format(info["learner_loss"]))
                        print("cost: {}".format(env.cost))
                        print("budget remain: {}".format(env.budget - env.cost))

                        # Update metrics
                        cumulative_reward = cumulative_reward + reward
                        total_duration = total_duration + info["duration"]

                        if done:
                            utils.export_csv(
                                exp_id=exp_id,
                                scheduler_name=scheduler_name,
                                env_name=env_name, 
                                algo_name=algo_name, 
                                csv_name="eval_{}".format(episode_id),
                                csv_file=csv_round
                            )
                            round_done = True
                            
                        state = next_state
                        mask = next_mask


if __name__ == "__main__":
    scheduler_name = "minions_eval"
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
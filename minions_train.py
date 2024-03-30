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

                # Record max cumulative reward
                max_cumulative_reward = -1e8
                min_cost = 1e8

                # Set up environment
                env = Environment(
                    scheduler_name=scheduler_name,
                    algo_name=algo_name,
                    env_name=env_name,
                    target_reward=config.envs[env_name]["max_reward"],
                    budget=config.envs[env_name]["budget"],
                    is_serverless=is_serverless
                )

                # Record csv
                csv_episode = [
                    [
                        "episode",
                        "cumulative_reward",
                        "loss",
                        "total_duration",
                        "total_cost"
                    ]
                ]

                # Start training
                for episode_id in range(config.max_episode_train):
                    state, mask, info = env.reset()
                    cumulative_reward = 0
                    total_duration = 0

                    state_history = []
                    mask_history = []
                    action_history = []
                    reward_history = []
                    value_history = []
                    log_prob_history = []

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
                        # print("")
                        # print("Running {}, algo {}, env {}, episode {}".format(scheduler_name, algo_name, env_name, episode_id))
                        # print("round: {}".format(info["round"]))
                        # print("duration: {}".format(info["duration"]))
                        # print("action: {}".format(action_item))
                        # print("reward: {}".format(reward))
                        # print("learner_time: {}".format(info["learner_time"]))
                        # print("actor_time: {}".format(info["actor_time"]))
                        # print("eval_reward: {}".format(info["eval_reward"]))
                        # print("learner_loss: {}".format(info["learner_loss"]))
                        # print("cost: {}".format(env.cost))
                        # print("budget remain: {}".format(env.budget - env.cost))

                        # Record trajectory
                        state = state.detach()
                        mask = mask.detach()
                        action = action.detach()
                        value_pred = value_pred.detach()
                        log_prob = log_prob.detach()

                        state_history.append(state)
                        mask_history.append(mask)
                        action_history.append(action)
                        reward_history.append(reward)
                        value_history.append(value_pred)
                        log_prob_history.append(log_prob)

                        # Update metrics
                        cumulative_reward = cumulative_reward + reward
                        total_duration = total_duration + info["duration"]

                        if done:
                            # Concatenate trajectories
                            state_history = torch.cat(state_history, dim=0)
                            mask_history = torch.cat(mask_history, dim=0)
                            action_history = torch.cat(action_history, dim=0)
                            value_history = torch.cat(value_history).squeeze()
                            log_prob_history = torch.cat(log_prob_history, dim=0)

                            loss = agent.update(
                                state_history=state_history,
                                mask_history=mask_history,
                                action_history=action_history,
                                reward_history=reward_history,
                                value_history=value_history,
                                log_prob_history=log_prob_history
                            )

                            if max_cumulative_reward < cumulative_reward:
                                max_cumulative_reward = cumulative_reward
                                max_cumulative_reward_episode = episode_id
                                agent.save(config.model_save_path + "max_cumulative_reward_{}_{}.pth".format(env_name, algo_name))

                                utils.export_csv(
                                    exp_id=exp_id,
                                    scheduler_name=scheduler_name,
                                    env_name=env_name, 
                                    algo_name=algo_name, 
                                    csv_name="round_max_cumulative_reward",
                                    csv_file=csv_round
                                )
                            
                            if min_cost > env.cost:
                                min_cost = env.cost
                                min_cost_episode = episode_id
                                agent.save(config.model_save_path + "min_cost_{}_{}.pth".format(env_name, algo_name))

                                utils.export_csv(
                                    exp_id=exp_id,
                                    scheduler_name=scheduler_name,
                                    env_name=env_name, 
                                    algo_name=algo_name, 
                                    csv_name="round_min_cost",
                                    csv_file=csv_round
                                )

                            csv_episode.append(
                                [
                                    episode_id,
                                    cumulative_reward,
                                    loss,
                                    total_duration,
                                    env.cost
                                ]
                            )
                            if (episode_id + 1) % 100 == 0:
                                utils.export_csv(
                                    exp_id=exp_id,
                                    scheduler_name=scheduler_name,
                                    env_name=env_name, 
                                    algo_name=algo_name, 
                                    csv_name="episode_{}".format(episode_id),
                                    csv_file=csv_episode
                                ) 

                            round_done = True
                            
                            print("")
                            print("**********")
                            print("")
                            print("Running {}, algo {}, env {}".format(scheduler_name, algo_name, env_name))
                            print("Exp {}, episode {} finished in {} rounds".format(exp_id, episode_id, info["round"]))
                            print("Cumulative reward: {}".format(cumulative_reward))
                            print("Loss: {}".format(loss))
                            print("Total duration: {}".format(total_duration))
                            print("Cost: {}".format(env.cost))
                            print("Current max cumulative reward: {}, observed at episode {}".format(max_cumulative_reward, max_cumulative_reward_episode))
                            print("Current min cost {}, observed at episode {}".format(min_cost, min_cost_episode))
                            print("")
                            print("**********")
                            print("**********")
                            print("**********")
                            print("")

                        state = next_state
                        mask = next_mask

                # Export to csv files
                utils.export_csv(
                    exp_id=exp_id,
                    scheduler_name=scheduler_name,
                    env_name=env_name, 
                    algo_name=algo_name, 
                    csv_name="episode",
                    csv_file=csv_episode
                )


if __name__ == "__main__":
    scheduler_name = "minions_train"
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
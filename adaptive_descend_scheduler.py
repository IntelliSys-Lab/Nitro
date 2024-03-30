import ray
from ray.rllib.algorithms import ppo, sac, a2c, pg, dqn, ddpg
import csv
import numpy as np
import time
import collections
import config
import utils


def experiment():
    scheduler_name = "adaptive_descend"

    for exp_id in range(config.num_exp):
        for algo_name in config.algos:
            for env_name in config.envs.keys():
                if algo_name == "ppo":
                    trainer_config = ppo.PPOConfig()
                elif algo_name == "sac":
                    trainer_config = sac.SACConfig()
                elif algo_name == "a2c":
                    trainer_config = a2c.A2CConfig()
                elif algo_name == "pg":
                    trainer_config = pg.PGConfig()
                elif algo_name == "dqn":
                    trainer_config = dqn.DQNConfig()
                elif algo_name == "ddpg":
                    trainer_config = ddpg.DDPGConfig()

                # Record csv
                csv_trend = [
                    [
                        "timestamp",
                        "iteration", 
                        "num_actor",
                        "learner_time", 
                        "actor_time",
                        "reward",
                        "loss",
                        "reward_ratio",
                    ]
                ]

                # Init trainer
                num_rollout_workers = config.num_min_rollout_workers
                train_batch_size = num_rollout_workers * config.rollout_fragment_length
                trainer = utils.make_trainer(
                    trainer_config=trainer_config,
                    framework=config.framework,
                    env_name=env_name,
                    num_gpus_for_local_worker=config.num_min_gpus_for_local_worker,
                    num_cpus_for_local_worker=config.num_min_cpus_for_local_worker,
                    num_cpus_per_worker=config.num_min_cpus_per_worker,
                    num_gpus_per_worker=config.num_min_gpus_per_worker,
                    train_batch_size=train_batch_size,
                    sgd_minibatch_size=train_batch_size,
                    rollout_fragment_length=config.rollout_fragment_length,
                    num_rollout_workers=num_rollout_workers,
                    num_envs_per_worker=config.num_min_envs_per_worker,
                )

                total_learner_time = 0
                total_actor_time = 0
                cur_iter = 0
                min_reward = config.envs[env_name]["min_reward"]
                max_reward = config.envs[env_name]["max_reward"]
                cur_reward = min_reward
                reward_window = collections.deque(maxlen=config.reward_window_size)
                bound_reward = True

                # Train
                start_time = time.time()
                while cur_iter < config.iteration:
                    reward_window.append(cur_reward)

                    # Adaptive rate
                    reward_ratio = utils.scale(
                        x=cur_reward, 
                        src=[min_reward, max_reward],
                        dst=[0, 1]
                    )
                    
                    old_num_rollout_workers = trainer.workers._worker_manager.num_actors()
                    num_rollout_workers = int((1-reward_ratio) * (config.num_max_rollout_workers - config.num_min_rollout_workers) + config.num_min_rollout_workers) + 1
                    num_rollout_workers = np.clip(num_rollout_workers, config.num_min_rollout_workers, config.num_max_rollout_workers)
                    train_batch_size = num_rollout_workers * config.rollout_fragment_length

                    # Adjust worker set
                    if num_rollout_workers > old_num_rollout_workers:
                        trainer.workers.add_workers(
                            num_workers=(num_rollout_workers - old_num_rollout_workers),
                            validate=True
                        )
                    elif num_rollout_workers < old_num_rollout_workers:
                        trainer.workers.remove_workers(
                            num_workers=(old_num_rollout_workers - num_rollout_workers),
                        )
                    # Update config
                    trainer.config['train_batch_size'] = train_batch_size
                    trainer.num_rollout_workers = num_rollout_workers
                    trainer.train_batch_size = train_batch_size
                    trainer.sgd_minibatch_size = train_batch_size
                    trainer.workers.sync_weights()

                    # Train one round
                    train_results = trainer.train()

                    # Collect results
                    # print(train_results)
                    num_episode = train_results["episodes_this_iter"]
                    num_timstep = train_results["num_steps_trained_this_iter"]
                    iter_time = train_results["time_this_iter_s"]
                    learner_time = train_results["timers"]["learn_time_ms"]/1000
                    actor_time = abs(iter_time - learner_time)
                    episode_reward = train_results["hist_stats"]["episode_reward"]
                    if len(episode_reward) == 0:
                        reward = np.nan
                    else:
                        if config.reward_norm:
                            reward = np.mean(utils.remove_outliers(episode_reward))
                        else:
                            reward = np.mean(episode_reward)
                    if algo_name == "ppo":
                        loss = train_results["info"]['learner']['default_policy']['learner_stats']['total_loss']
                    elif algo_name == "sac":
                        loss = train_results["info"]['learner']['default_policy']['mean_td_error']
                    elif algo_name == "a2c":
                        loss = train_results["info"]['learner']['default_policy']['learner_stats']['policy_loss'] + train_results["info"]['learner']['default_policy']['learner_stats']['vf_loss']
                    elif algo_name == "pg":
                        loss = train_results["info"]['learner']['default_policy']['learner_stats']['policy_loss']
                    elif algo_name == "dqn":
                        loss = train_results["info"]['learner']['default_policy']['mean_td_error']
                    elif algo_name == "ddpg":
                        loss = train_results["info"]['learner']['default_policy']['mean_td_error']
                    
                    total_learner_time = total_learner_time + learner_time
                    total_actor_time = total_actor_time + num_rollout_workers * actor_time
                    cur_time = time.time() - start_time
                    cur_iter = cur_iter + 1
                    if reward is not np.nan:
                        cur_reward = reward

                    print("**********")
                    print("Exp id: {}".format(exp_id))
                    print("Scheduler: {}".format(scheduler_name))
                    print("Environment: {}".format(env_name))
                    print("Algorithm: {}".format(algo_name))
                    print("Timestamp: {}".format(cur_time))
                    print("Iteration: {}".format(cur_iter))
                    print("Episodes: {}".format(num_episode))
                    print("Timesteps: {}".format(num_timstep))
                    print("Iteration time: {}".format(iter_time))
                    print("Num of actors: {}".format(num_rollout_workers))
                    print("Learner time: {}".format(learner_time))
                    print("Actor time: {}".format(actor_time))
                    print("Reward: {}".format(reward))
                    print("Loss: {}".format(loss))
                    print("Reward ratio: {}".format(reward_ratio))
                    print("**********")
                    print("")

                    # Log csv files
                    csv_trend.append(
                        [
                            cur_time,
                            cur_iter,
                            num_rollout_workers,
                            learner_time, 
                            actor_time, 
                            reward, 
                            loss,
                            reward_ratio
                        ]
                    )

                    if bound_reward and np.mean(reward_window) >= max_reward:
                        utils.export_csv(
                            exp_id=exp_id,
                            scheduler_name=scheduler_name,
                            env_name=env_name, 
                            algo_name=algo_name, 
                            csv_name="b",
                            csv_file=csv_trend
                        )
                        bound_reward = False

                total_time = time.time() - start_time

                # Export to csv files
                utils.export_csv(
                    exp_id=exp_id,
                    scheduler_name=scheduler_name,
                    env_name=env_name, 
                    algo_name=algo_name, 
                    csv_name="t",
                    csv_file=csv_trend
                )

                if bound_reward:
                    utils.export_csv(
                        exp_id=exp_id,
                        scheduler_name=scheduler_name,
                        env_name=env_name, 
                        algo_name=algo_name, 
                        csv_name="b",
                        csv_file=csv_trend
                    )
                    bound_reward = False

                trainer.stop()

if __name__ == "__main__":
    # Run experiment
    ray.init(
        num_cpus=config.num_cpus,
        num_gpus=config.num_gpus,
        log_to_driver=False,
    )
    experiment()
    ray.shutdown()

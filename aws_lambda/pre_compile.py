from serverless_actor import ServerlessActor
import config


def pre_compile():
    # Ray and environemnts
    redis_host = "localhost"
    redis_port = 6380
    redis_password = "password"
    
    for algo_name in config.algos:
        for env_name in config.envs.keys():
            actor = ServerlessActor(
                redis_host=redis_host,
                redis_port=redis_port,
                redis_password=redis_password,
                algo_name=algo_name,
                env_name=env_name,
                num_envs_per_worker=config.num_envs_per_worker,
                rollout_fragment_length=config.envs[env_name]['rollout_fragment_length'],
            )
            print("")
            print("algo_name: {}, env_name: {}".format(algo_name, env_name))
            print(actor.sample())
    

if __name__ == '__main__':
    pre_compile()

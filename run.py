import ray
import config
import fixed_scheduler, ascend_scheduler, descend_scheduler, adaptive_ascend_scheduler, adaptive_descend_scheduler


if __name__ == "__main__":
    # Init ray cluster
    ray.init(
        num_cpus=config.num_cpus,
        num_gpus=config.num_gpus,
        log_to_driver=False,
    )
    
    # Fixed
    # fixed_scheduler.experiment(plan=[1, 2, 4, 8, 16, 32])

    # Ascend
    # ascend_scheduler.experiment(plan=[5, 10])

    # Descend
    # descend_scheduler.experiment(plan=[5, 10])

    # Adaptive ascend
    adaptive_ascend_scheduler.experiment()

    # Adaptive descend
    # adaptive_descend_scheduler.experiment()

    # End
    ray.shutdown()
    
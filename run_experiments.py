import ray
import logging
import boost_min, boost_max, boost_kungfu, boost_gns, boost_gns_rev, boost_rd


if __name__ == "__main__":
    
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

    print("")
    print("Start training...")
    print("")
    for baseline in [boost_min, boost_max, boost_kungfu, boost_gns, boost_gns_rev, boost_rd]:
        scheduler_name = baseline.__name__
        is_serverless = False
        
        print("Running {}, is serverless? {}".format(scheduler_name, is_serverless))
        baseline.experiment(
            scheduler_name=scheduler_name,
            is_serverless=is_serverless
        )
        print("")
        print("{} training finished!".format(scheduler_name))
        print("")

    ray.shutdown()
    print("**********")
    print("**********")
    print("**********")
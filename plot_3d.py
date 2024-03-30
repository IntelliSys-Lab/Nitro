import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
import config
import utils
import pandas as pd
from scipy.interpolate import griddata


def plot_3d(
    csv_path,
    save_path, 
):
    # Read data from CSV
    df = pd.read_csv(csv_path)
    offset_1 = df['offset_1']
    offset_2 = df['offset_2']
    episode_reward_mean = df['episode_reward_mean'].astype('int')

    # Interpolate grid
    x1 = np.linspace(offset_1.min(), offset_1.max(), len(offset_1))
    y1 = np.linspace(offset_2.min(), offset_2.max(), len(offset_2))
    X, Y = np.meshgrid(x1, y1)
    Z = griddata((offset_1, offset_2), episode_reward_mean, (X, Y), method='cubic')

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(X, Y, Z, 
        rstride=1, 
        cstride=1, 
        cmap=cm.coolwarm,
        linewidth=0, 
        antialiased=False,
    )

    fig.colorbar(surf, shrink=0.5, aspect=5)

    # Save plot
    fig.savefig(
        save_path, 
        dpi=1000,
        bbox_inches='tight', 
    )


if __name__ == '__main__':
    utils.mkdir(config.plot_path)

    scheduler_names = [
        # "eval_reward_surface_min_convex",
        # "eval_reward_surface_max_convex",
        # "eval_reward_surface_5",
        # "eval_reward_surface_10",
        # "eval_reward_surface_15",
        # "eval_boost_min_convex",
        # "eval_boost_max_convex",
        # "eval_boost_5",
        # "eval_boost_10",
        # "eval_boost_15",
        "eval_server_startup",
    ]
    save_types = ["pdf", "png"]

    for algo_name in config.algos:
        for env_name in config.envs.keys():
            for scheduler_name in scheduler_names:
                for save_type in save_types:
                    csv_path = "{}/{}~{}~{}~surface.csv".format(config.log_path, scheduler_name, env_name, algo_name)
                    save_path = "{}/{}~{}~{}~surface.{}".format(config.plot_path, scheduler_name, env_name, algo_name, save_type)
                    plot_3d(
                        csv_path=csv_path,
                        save_path=save_path, 
                    )

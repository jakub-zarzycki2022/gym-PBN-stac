"""Utility to log WNT5A stuff to wandb
"""

import gym
import gym_PBN
from gym_PBN.utils.eval import compute_ssd_hist
import wandb
from pathlib import Path
from gym_PBN.envs.pbn_target import PBNTargetEnv
from gym_PBN.envs.bittner.utils import spawn

RUN_NAME = "Bittner-28"

if __name__ == "__main__":
    wandb.init(project="pbn-rl", entity="uos-plccn", name=RUN_NAME, group="PBN SSDs")

    # env = PBNTargetEnv(graph, goal_config, "Bittner-70")
    env = gym.make("gym-PBN/Bittner-28-v0")

    ssd, _ = compute_ssd_hist(env)
    wandb.log({"WNT5A Off": ssd.iloc[0][0], "WNT5A On": ssd.iloc[1][0]})

    wandb.finish()

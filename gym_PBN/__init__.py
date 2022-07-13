from gym.envs.registration import register

register(id="gym-PBN/PBN-v0", entry_point="gym_PBN.envs:PBNEnv")

register(id="gym-PBN/PBN-target-v0", entry_point="gym_PBN.envs:PBNTargetEnv")

register(
    id="gym-PBN/Bittner-28-v0",
    entry_point="gym_PBN.envs:Bittner28",
    nondeterministic=True,
    max_episode_steps=100,
)

register(
    id="gym-PBN/Bittner-70-v0",
    entry_point="gym_PBN.envs:Bittner70",
    nondeterministic=True,
    max_episode_steps=100,
)

register(
    id="gym-PBN/Bittner-100-v0",
    entry_point="gym_PBN.envs:Bittner100",
    nondeterministic=True,
    max_episode_steps=100,
)

register(
    id="gym-PBN/Bittner-200-v0",
    entry_point="gym_PBN.envs:Bittner200",
    nondeterministic=True,
    max_episode_steps=100,
)

register(id="gym-PBN/PBN-sampled-data-v0", entry_point="gym_PBN.envs:PBNSampledDataEnv")

register(
    id="gym-PBN/PBN-self-triggering-v0", entry_point="gym_PBN.envs:PBNSelfTriggeringEnv"
)

register(id="gym-PBN/PBCN-v0", entry_point="gym_PBN.envs:PBCNEnv")

register(
    id="gym-PBN/PBCN-sampled-data-v0", entry_point="gym_PBN.envs:PBCNSampledDataEnv"
)

register(
    id="gym-PBN/PBCN-self-triggering-v0",
    entry_point="gym_PBN.envs:PBCNSelfTriggeringEnv",
)

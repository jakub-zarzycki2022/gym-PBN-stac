from gym.envs.registration import register

register(id="PBN-v0", entry_point="gym_PBN.envs:PBNEnv")

register(id="PBN-sampled-data-v0", entry_point="gym_PBN.envs:PBNSampledDataEnv")

register(id="PBN-self-triggering-v0", entry_point="gym_PBN.envs:PBNSelfTriggeringEnv")

register(id="PBCN-v0", entry_point="gym_PBN.envs:PBCNEnv")

register(id="PBCN-sampled-data-v0", entry_point="gym_PBN.envs:PBCNSampledDataEnv")

register(id="PBCN-self-triggering-v0", entry_point="gym_PBN.envs:PBCNSelfTriggeringEnv")

from gym.envs.registration import register

register(
    id='PBN-v0',
    entry_point='gym_PBN.envs:PBNEnv'
)

register(
    id='PBCN-v0',
    entry_point='gym_PBN.envs:PBCNEnv'
)

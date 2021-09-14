import gym

env = gym.make("gym_PBN:PBCN-v0", logic_func_data=(
    ["u", "x1", "x2", "x3", "x4"],
    [
        [],
        [("not x2 and not x4", 1)],
        [("not x4 and not u and (x2 or x3)", 1)],
        [("not x2 and not x4 and x1", 0.7), ("False", 0.3)],
        [("not x2 and not x3", 1)]
    ]
), goal_config={
    "all_attractors": [{(0, 0, 0, 1)}, {(0, 1, 0, 0)}],
    "target": {(0, 0, 0, 1)}
})


obs = env.reset()
for i in range(10):
    obs, reward, done, info = env.step([1])
    print(obs, reward, done)

    if done:
        break

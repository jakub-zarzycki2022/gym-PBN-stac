import gym


def iterate_through_env(env, action, steps):
    """
    Once you have an and an action, all the function calls you need to iterate through it and
    interact with it.
    """
    obs = env.reset()
    for i in range(steps):
        obs, reward, done, info = env.step(action)
        print(obs, reward, done)

        if done:
            break


def test_example_1():
    """
    How to make a custom PBCN if you have logic functions from, say
    a paper.
    """
    env = gym.make(
        "gym_PBN/PBCN-v0",
        logic_func_data=(
            ["u", "x1", "x2", "x3", "x4"],
            [
                [],
                [("not x2 and not x4", 1)],
                [("not x4 and not u and (x2 or x3)", 1)],
                [("not x2 and not x4 and x1", 0.7), ("False", 0.3)],
                [("not x2 and not x3", 1)],
            ],
        ),
        goal_config={
            "all_attractors": [{(0, 0, 0, 1)}, {(0, 1, 0, 0)}],
            "target": {(0, 0, 0, 1)},
        },
    )

    iterate_through_env(env, [1], 10)


def test_example_2():
    """
    Loading and iterating through a Bittner PBN, which are our only pre-packaged PBNs.
    You will need logic functions and instantiate a gym_PBN/PBN-v0 or gym_PBN/PBCN-v0 env like above.
    """
    env = gym.make("gym-PBN/Bittner-200-v0")
    iterate_through_env(env, 0, 11)

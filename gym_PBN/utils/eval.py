from gym_PBN.envs.pbn_target import PBNTargetEnv
from typing import Iterable
import itertools

import pandas as pd
import numpy as np

from tqdm import tqdm


def _bit_seq_to_str(seq: Iterable[int]) -> str:
    return "".join([str(i) for i in seq])


def compute_ssd_hist(
    env: PBNTargetEnv,
    model: object = None,
    iters: int = 1_200_000,
    resets: int = 300,
    bit_flip_prob: float = 0.01,
) -> pd.DataFrame:
    SSD_N = iters  # Number of experiences to sample for the SSD calculation
    SSD_RESETS = resets
    BIT_FLIP_PROB = bit_flip_prob

    assert (
        bit_flip_prob >= 0 and bit_flip_prob <= 1
    ), "Invalid Bit Flip Probability value."
    assert SSD_RESETS > 0, "Invalid resets value."
    assert SSD_N > 0, "Invalid iterations value."
    assert SSD_N // SSD_RESETS, "Resets does not divide the iterations."

    g = len(env.target_nodes)
    ssd = np.zeros((2**g, SSD_RESETS), dtype=np.float32)

    total_iters = 0
    for i in tqdm(range(SSD_RESETS), desc=f"SSD run for {env.name}"):
        sub_ssd = np.zeros(2**g, dtype=np.float32)
        env.reset()

        for _ in range(SSD_N // SSD_RESETS):
            total_iters += 1

            state = env.render()
            # Convert relevant part of state to binary string, then parse it as an int to get the bucket index.
            bucket = env.render(mode="target_idx")
            sub_ssd[bucket] += 1

            if not model:  # Control the environment
                flip = np.random.rand(len(state)) < BIT_FLIP_PROB
                for j in range(len(state)):
                    if flip[j]:
                        env.graph.flipNode(j)
                env.step(action=0)
            else:
                action = model.predict(state, deterministic=True)
                if type(action) == tuple:
                    action = action[0]
                env.step(action=action)

        ssd[:, i] = sub_ssd

    ssd = np.mean(ssd, axis=1)
    ssd /= SSD_N // SSD_RESETS  # Normalize
    ret = ssd

    states = list(map(_bit_seq_to_str, itertools.product([0, 1], repeat=g)))
    ret = pd.DataFrame(list(ssd), index=states, columns=["Value"])

    return ret

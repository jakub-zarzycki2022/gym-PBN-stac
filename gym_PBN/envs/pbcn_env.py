from typing import Tuple, Union

import networkx as nx
from gym.spaces import Discrete, MultiBinary
from gym_PBN.types import GYM_STEP_RETURN, REWARD, STATE, TERMINATED
from gym_PBN.utils import booleanize

from .common.pbcn import PBCN
from .pbn_env import PBNEnv


class PBCNEnv(PBNEnv):
    metadata = {"render.modes": ["human", "PBN", "STG", "funcs", "idx", "float"]}

    def __init__(
        self,
        PBN_data=[],
        logic_func_data=None,
        name: str = None,
        goal_config: dict = None,
        reward_config: dict = None,
    ):
        super().__init__(PBN_data, logic_func_data, name, goal_config, reward_config)

        # Switch to PBCN
        self.PBN = PBCN(PBN_data, logic_func_data)

        # Update Gym
        self.observation_space = MultiBinary(self.PBN.N)
        self.observation_space.dtype = bool
        self.action_space = MultiBinary(self.PBN.M)
        self.action_space.dtype = bool
        self.discrete_action_space = Discrete(2**self.action_space.n)

    def _get_reward(self, observation: STATE) -> Tuple[REWARD, TERMINATED]:
        reward, done = 0, False
        observation_tuple = tuple(observation)

        if observation_tuple in self.target:
            reward += self.successful_reward
            done = True
        else:
            attractors_matched = sum(
                observation_tuple in attractor for attractor in self.all_attractors
            )
            reward -= self.wrong_attractor_cost * attractors_matched

        return reward, done

    def step(self, action: Union[Tuple[int], int]) -> GYM_STEP_RETURN:
        if type(action) is int:
            action = booleanize(action, self.action_space.n)

        if not self.action_space.contains(action):
            raise Exception(f"Invalid action {action}, not in action space.")

        self.PBN.apply_control(action)

        self.PBN.step()

        observation = self.PBN.state
        reward, done = self._get_reward(observation)
        info = {"observation_idx": self._state_to_idx(observation)}

        return observation, reward, done, info

    def compute_attractors(self):
        attractor_sets = []
        for action in self.PBN.control_actions:
            print(f"Computing attractors for action {action}...")
            self.PBN.apply_control(action)
            STG = self.render(mode="STG", no_cache=True)
            generator = nx.algorithms.components.attracting_components(STG)
            attractors = self._nx_attractors_to_tuples(list(generator))
            attractor_sets.append((action, attractors))
        return attractor_sets

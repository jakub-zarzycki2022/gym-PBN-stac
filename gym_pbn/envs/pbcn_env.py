from typing import Tuple

from gym.spaces import MultiBinary
import networkx as nx

from .pbn_env import PBNEnv
from .common.pbcn import PBCN
from gym_PBN.types import GYM_STEP_RETURN, REWARD, STATE, TERMINATED


class PBCNEnv(PBNEnv):
    metadata = {
        "render.modes": ["cli", "PBN", "STG", "funcs"]
    }

    def __init__(self, PBN_data = [], logic_func_data = None, goal_config: dict = None, reward_config: dict = None):
        super().__init__(PBN_data, logic_func_data, goal_config, reward_config)

        # Switch to PBCN
        self.PBN = PBCN(PBN_data, logic_func_data)

        # Update Gym
        self.observation_space = MultiBinary(self.PBN.N)
        self.observation_space.dtype = bool
        self.action_space = MultiBinary(self.PBN.M)
        self.action_space.dtype = bool

    def _get_reward(self, observation: STATE) -> Tuple[REWARD, TERMINATED]:
        reward, done = 0, False
        observation_tuple = tuple(observation)

        if observation_tuple in self.target:
            reward += self.successful_reward
            done = True
        else:
            attractors_matched = sum(observation_tuple in attractor for attractor in self.all_attractors)
            reward -= self.wrong_attractor_cost * attractors_matched

        return reward, done

    def step(self, action: Tuple[int]) -> GYM_STEP_RETURN:
        if not self.action_space.contains(action):
            raise Exception(f"Invalid action {action}, not in action space.")
        
        self.PBN.apply_control(action)

        self.PBN.step()

        observation = self.PBN.state
        reward, done = self._get_reward(observation)
        info = {}

        return observation, reward, done, info

    def compute_attractors(self):
        attractor_sets = []
        for action in self.PBN.control_actions:
            self.PBN.apply_control(action)
            STG = self.render(mode="STG", no_cache=True)
            generator = nx.algorithms.components.attracting_components(STG)
            attractors = self._nx_attractors_to_tuples(list(generator))
            attractor_sets.append((action, attractors))
        return attractor_sets

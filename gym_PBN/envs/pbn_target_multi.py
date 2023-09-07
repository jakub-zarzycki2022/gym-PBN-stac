import random
from collections import defaultdict
from pathlib import Path
from typing import List, Set, Tuple, Union
import pickle as pkl

import gymnasium as gym
import networkx as nx
import numpy as np
import torch
from gymnasium.spaces import Discrete, MultiBinary, MultiDiscrete
from gym_PBN.types import GYM_STEP_RETURN, REWARD, STATE, TERMINATED, TRUNCATED

from .bittner import base, utils
from .bittner.base import findAttractors

from gym_PBN.utils.get_attractors_from_cabean import get_attractors


def state_equals(state1, state2):
    for i in range(len(state2)):
        if state1[i] != state2[i]:
            return False
    return True


class PBNTargetMultiEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "dict", "PBN", "STG", "idx", "float", "target"]
    }

    def __init__(
        self,
        graph: base.Graph,
        goal_config: dict,
        render_mode: str = None,
        render_no_cache: bool = False,
        name: str = None,
        reward_config: dict = None,
        end_episode_on_success: bool = False,
    ):
        self.target = None
        self.graph = graph

        self.end_episode_on_success = end_episode_on_success

        if "horizon" in goal_config.keys():
            print("setting from config")
            self.horizon = goal_config["horizon"]
        else:
            print("setting 100")
            self.horizon = 100

        # Gym
        self.observation_space = MultiBinary(self.graph.N)
        # intervention nodes + no action
        print("\nhello\n")
        self.action_space = MultiDiscrete(self.graph.N + 1)
        self.name = name
        self.render_mode = render_mode
        self.render_no_cache = render_no_cache

        # State
        self.n_steps = 0

        self.visited_states = defaultdict(int)

        self.all_attractors = []
        self.non_attractors = set()
        self.counter = 0

    def _seed(self, seed: int = None):
        np.random.seed(seed)
        random.seed(seed)

    def _check_config(
        self,
        config: dict,
        _type: str,
        required_keys: Set[str],
        default_values: dict = None,
    ) -> dict:
        """Small utility function to validate an environment config.

        Args:
            config (dict): The config to validate.
            _type (str): The type of config this is about. Just needs to be a semantically rich string for the exception output.
            required_keys (Set[str]): The mandatory keys that need to be in the config.
            default_values (dict, optional): The default values for the config should it be empty. Defaults to None.

        Raises:
            ValueError: Thrown when some or all of the required keys are missing in the given config values.

        Returns:
            dict: The config after it has been checked and initialized to default values if it was empty to begin with.
        """
        if config:
            missing_keys = required_keys - set(config.keys())
            if len(missing_keys) > 1:  # If any of the required keys are missing
                raise ValueError(
                    f"Invalid {_type} config provided. The following required values are missing: {', '.join(missing_keys)}."
                )
        else:
            config = default_values

        return config

    def step(self, actions, force=False):

        #print(f"for state {self.get_state()} got action {action}")
        # if not self.action_space.contains(action):
        #     raise Exception(f"Invalid action {action}, not in action space.")

        self.n_steps += 1
        #print(f"action = {action}")

        #s = self.render()
        #print(self.is_attracting_state(s), s, action)
        for action in actions:
            if action != 0:  # Action 0 is taking no action.
                self.graph.flipNode(action - 1)

        self.graph.step()
        while not force and not self.is_attracting_state(self.graph.getState().values()):
            self.graph.step()

        observation = self.graph.getState()
        reward, terminated, truncated = self._get_reward(observation, actions)
        info = {
            "observation_idx": self._state_to_idx(observation),
            "observation_dict": observation,
        }

        return self.get_state(), reward, terminated, truncated, info

    def _to_map(self, state):
        getIDs = getattr(self.graph, "getIDs", None)
        if getIDs is not None and type(state) is not dict:
            ids = getIDs()
            state = dict(zip(ids, state))
        return state

    def in_target(self, observation):
        if self.target is None:
            raise ValueError("Target should have been initialized during env.reset()")

        for a_state in self.target:
            for i in range(len(observation)):
                if a_state[i] == '*':
                    continue
                if a_state[i] != observation[i]:
                    break
            else:
                return True
        return False

    def _get_reward(self, observation: STATE, actions) -> Tuple[REWARD, TERMINATED, TRUNCATED]:
        assert isinstance(actions, list)
        """The Reward function.

        Args:
            observation (STATE): The next state observed as part of the action.
            action (int): The action taken.

        Returns:
            Tuple[REWARD, TERMINATED, TRUNCATED]: Tuple of the reward and the environment done status.
        """
        reward, terminated = 0, False
        observation = tuple(observation.values())

        if self.in_target(observation):
            reward += 1000
            terminated = True

        reward -= 1 * len(actions)

        truncated = self.n_steps == self.horizon
        return reward, terminated, truncated

    def reset(self, seed: int = None, options: dict = None):
        """Reset the environment. Initialise it to a random state, or to a certain state."""
        if seed:
            self._seed(seed)

        state_attractor, target_attractor = random.sample(self.all_attractors, 2)
        state = list(random.choice(state_attractor))
        target = list(random.choice(target_attractor))
        for i in range(len(state)):
            if state[i] == "*":
                state[i] = random.randint(0, 1)
            if target[i] == "*":
                target[i] = random.randint(0, 1)

        self.graph.setState(state)

        self.n_steps = 0
        observation = self.graph.getState()
        info = {
            "observation_idx": self._state_to_idx(observation),
            "observation_dict": observation,
        }

        self.target = target_attractor
        return (tuple(state), tuple(target)), info

    def get_state(self):
        return np.array(list(self.graph.getState().values()))

    def setTarget(self, target):
        self.target = target

    def render(self, mode=None):
        mode = self.render_mode if not mode else mode

        if mode == "human":
            return self.get_state()
        if mode == "dict":
            return self.graph.getState()
        elif mode == "PBN":
            return self.graph.printGraph()
        elif mode == "STG":
            return self.graph.genSTG()
        elif mode == "idx":
            return self._state_to_idx(self.graph.getState())
        elif mode == "float":
            return [float(x) for x in self.graph.getState()]
        elif mode == "target":
            state = self.graph.getState()
            return [state[node] for node in self.target_nodes]
        elif mode == "target_idx":
            target_state = self.render(mode="target")
            return self._state_to_idx(target_state)

    # AM: Added to hadle the problem of render() method not accepting the keyword argument 'mode'.
    def getTargetIdx(self):
        state = self.graph.getState()
        target_state = [state[node] for node in self.target_nodes]
        return self._state_to_idx(target_state)

    def _state_to_idx(self, state: STATE):
        if type(state) is dict:
            state = list(state.values())
        return int("".join([str(x) for x in state]), 2)

    def compute_attractors(self):
        print("Computing attractors...")
        STG = self.render(mode="STG")
        generator = nx.algorithms.components.attracting_components(STG)
        return self._nx_attractors_to_tuples(list(generator))

    def _nx_attractors_to_tuples(self, attractors):
        return [
            set(
                [
                    tuple([int(x) for x in state.lstrip("[").rstrip("]").split()])
                    for state in list(attractor)
                ]
            )
            for attractor in attractors
        ]

    def close(self):
        """Close out the environment and make sure everything is garbage collected."""
        del self.graph


class BittnerMulti70(PBNTargetMultiEnv):
    predictor_sets_path = Path(__file__).parent / "bittner" / "data"
    genedata = predictor_sets_path / "genedata.xls"

    includeIDs = [234237, 324901, 759948, 25485, 266361, 108208, 130057]

    N = 70
    NAME = "Bittner-70"

    def __init__(
        self,
        render_mode: str = "human",
        render_no_cache: bool = False,
        name: str = None,
        horizon: int = 69,
        reward_config: dict = None,
        end_episode_on_success: bool = True,
    ):
        print(f"its me, bittner-{self.N}")
        if not name:
            name = self.NAME

        graph = utils.spawn(
            file=self.genedata,
            total_genes=self.N,
            include_ids=self.includeIDs,
            bin_method="median",
            n_predictors=3,
            predictor_sets_path=self.predictor_sets_path,
        )

        goal_config = {
            "target_nodes": [234237, 324901, 759948, 25485, 266361, 108208, 130057],
            "intervene_on": [234237],
            "target_node_values": ((0, 0, 0, 0, 0, 0, 0),),
            "undesired_node_values": tuple(),
            "horizon": horizon,
        }
        super().__init__(
            graph,
            goal_config,
            render_mode,
            render_no_cache,
            name,
            reward_config,
            end_episode_on_success,
        )


class BittnerMulti100(BittnerMulti70):
    N = 100
    NAME = "Bittner-100"


class Bittner200(BittnerMulti70):
    N = 200
    NAME = "Bittner-200"


class BittnerMulti7(PBNTargetMultiEnv):
    predictor_sets_path = Path(__file__).parent / "bittner" / "data"
    genedata = predictor_sets_path / "genedata.xls"

    includeIDs = [234237, 324901, 759948, 25485, 266361, 108208, 130057]
    includeIDs = sorted(includeIDs)

    N = 7
    NAME = "Bittner-7"

    def __init__(
            self,
            render_mode: str = "human",
            render_no_cache: bool = False,
            name: str = None,
            horizon: int = 100,
            reward_config: dict = None,
            end_episode_on_success: bool = True,
    ):
        if not name:
            name = self.NAME

        print(f"initing {name}")

        graph = utils.spawn(
            file=self.genedata,
            total_genes=self.N,
            include_ids=self.includeIDs,
            bin_method="median",
            n_predictors=3,
            predictor_sets_path=self.predictor_sets_path,
        )

        goal_config = {
            "target_nodes": [234237, 324901, 759948, 25485, 266361, 108208, 130057],
            "intervene_on": [234237, 324901, 759948, 25485, 266361, 108208, 130057],
            "target_node_values": ((1, 1, 1, 1, 1, 1, 0),),
            "undesired_node_values": tuple(),
            "horizon": horizon,
        }
        super().__init__(
            graph,
            goal_config,
            render_mode,
            render_no_cache,
            name,
            reward_config,
            end_episode_on_success,
        )

        # its too big for PBN > 10
        if self.N < 11:
            stg = self.graph.genSTG()
            self.real_attractors = findAttractors(stg)
            print(f"real attractors are: {self.real_attractors}")

        self.all_attractors = get_attractors(self)

        print(self.all_attractors)

        self.target_nodes = sorted(self.includeIDs)
        self.target_node_values = self.all_attractors[-1]
        self.target_attractor = len(self.all_attractors) - 1  # last one

    def statistical_attractors(self):
        with open(f"data/attractors_{self.name}.pkl", 'r+b') as attractors:
            try:
                statistial_attractors = pkl.load(attractors)
                print("reusing old attractors")
            except:
                print(f"Calculating state statistics for N = {self.N}")
                print(f"it should take {10000} steps")
                state_log = defaultdict(int)

                for i in range(100):
                    #print(i)
                    _ = self.reset()
                    for j in range(1000):
                        state = tuple(self.render())
                        state_log[state] += 1
                        _ = self.step(0, force=True)

                states = sorted(state_log.items(), key=lambda kv: kv[1], reverse=True)

                statistial_attractors = [node for node, frequency in states[:4]]
                pkl.dump(statistial_attractors, file=attractors)
            return statistial_attractors

    def is_attracting_state(self, state):
        state = tuple(state)

        for attractor in self.all_attractors:
            for a_state in attractor:
                for i in range(len(state)):
                    if a_state[i] == '*':
                        continue
                    if a_state[i] != state[i]:
                        break
                else:
                    return True
        return False


class BittnerMulti10(BittnerMulti7):
    N = 10
    NAME = "BittnerMulti-10"


class BittnerMulti30(BittnerMulti7):
    N = 30
    NAME = "BittnerMulti-30"


class BittnerMulti28(BittnerMulti7):
    N = 28
    NAME = "BittnerMulti-28"
    def __init__(
            self,
            render_mode: str = "human",
            render_no_cache: bool = False,
            name: str = "Bittner-28",
            horizon: int = 100,
            reward_config: dict = None,
            end_episode_on_success: bool = False,
    ):
        includeIDs = [234237, 324901, 759948, 25485, 324700, 43129, 266361, 108208, 40764, 130057, 39781, 49665, 39159,
                      23185, 417218, 31251, 343072, 142076, 128100, 376725, 112500, 241530, 44563, 36950, 812276, 51018,
                      306013, 418105]

        includeIDs = sorted(includeIDs)

        self.includeIDs = includeIDs
        super().__init__()

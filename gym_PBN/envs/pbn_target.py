import random
from pathlib import Path
from typing import List, Set, Tuple, Union

import gym
import networkx as nx
import numpy as np
from gym.spaces import Discrete, MultiBinary
from gym_PBN.types import GYM_STEP_RETURN, REWARD, STATE, TERMINATED

from .bittner import base, utils


class PBNTargetEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "dict", "PBN", "STG", "idx", "float", "target"]
    }

    def __init__(
        self,
        graph: base.Graph,
        goal_config: dict,
        name: str = None,
        reward_config: dict = None,
        end_episode_on_success: bool = False,
    ):
        self.graph = graph

        # Goal configuration
        goal_config = self._check_config(
            goal_config,
            "goal",
            set(
                [
                    "target_nodes",
                    "target_node_values",
                    "undesired_node_values",
                    "intervene_on",
                ]
            ),
        )
        if goal_config is None:
            raise ValueError(
                "Target nodes, target values and intervention nodes need to be specified."
            )
        self.target_nodes = goal_config["target_nodes"]
        self.target_node_values = goal_config["target_node_values"]
        self.undesired_node_values = goal_config["undesired_node_values"]
        self.intervene_on = goal_config["intervene_on"]
        self.end_episode_on_success = end_episode_on_success

        if "horizon" in goal_config.keys():
            self.horizon = goal_config["horizon"]
        else:
            self.horizon = 11

        # Reward configuration
        reward_config = self._check_config(
            reward_config,
            "reward",
            set(["successful_reward", "wrong_attractor_cost", "action_cost"]),
            default_values={
                "successful_reward": 10,
                "wrong_attractor_cost": 2,
                "action_cost": 1,
            },
        )
        self.successful_reward = reward_config["successful_reward"]
        self.wrong_attractor_cost = reward_config["wrong_attractor_cost"]
        self.action_cost = reward_config["action_cost"]

        # Gym
        self.observation_space = MultiBinary(self.graph.N)
        # intervention nodes + no action
        self.action_space = Discrete(len(self.intervene_on) + 1)
        self.name = name

        # State
        self.n_steps = 0

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

    def step(self, action: int = 0) -> GYM_STEP_RETURN:
        """Transition the environment by 1 step. Optionally perform an action.

        Args:
            action (int, optional): The action to perform (1-indexed node to flip). Defaults to 0, meaning no action.

        Raises:
            Exception: When the action is outside the action space.

        Returns:
            GYM_STEP_RETURN: The typical Gym environment 4-item Tuple.\
                 Consists of the resulting environment state, the associated reward, the termination status and additional info.
        """
        if not self.action_space.contains(action):
            raise Exception(f"Invalid action {action}, not in action space.")

        self.n_steps += 1

        if action != 0:  # Action 0 is taking no action.
            _id = self.graph.getIDs().index(self.intervene_on[action - 1])
            self.graph.flipNode(_id)

        self.graph.step()

        observation = self.graph.getState()
        reward, done = self._get_reward(observation, action)
        info = {
            "observation_idx": self._state_to_idx(observation),
            "observation_dict": observation,
        }

        return self.get_state(), reward, done, info

    def _to_map(self, state):
        getIDs = getattr(self.graph, "getIDs", None)
        if getIDs is not None and type(state) is not dict:
            ids = getIDs()
            state = dict(zip(ids, state))
        return state

    def _get_reward(self, observation: STATE, action: int) -> Tuple[REWARD, TERMINATED]:
        """The Reward function.

        Args:
            observation (STATE): The next state observed as part of the action.
            action (int): The action taken.

        Returns:
            Tuple[REWARD, TERMINATED, bool]: Tuple of the reward and the environment done status.
        """
        reward, done = 0, False
        observation = self._to_map(observation)  # HACK Needed for some envs
        observation = tuple(
            [observation[x] for x in self.target_nodes]
        )  # Filter it down

        if observation in self.target_node_values:
            reward += self.successful_reward
            done = done or self.end_episode_on_success
        elif observation in self.undesired_node_values:
            reward -= self.wrong_attractor_cost
        else:
            reward -= self.successful_reward

        if action != 0:
            reward -= self.action_cost

        done = done or (self.end_episode_on_success and self.n_steps == self.horizon)

        return reward, done

    def reset(self, seed=None):
        """Reset the environment. Initialise it to a random state, or to a certain state."""
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        self.n_steps = 0
        self.graph.genRandState()
        return self.get_state()

    def set_state(self, state: Union[List[Union[int, bool]], np.ndarray, None]):
        return self.graph.setState(state)

    def get_state(self):
        return list(self.graph.getState().values())

    def render(self, mode="human", no_cache: bool = False):
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
        else:
            raise Exception(f'Unrecognised mode "{mode}"')

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


class Bittner28(PBNTargetEnv):
    predictor_sets_path = Path(__file__).parent / "bittner" / "data"
    genedata = predictor_sets_path / "genedata.xls"

    # fmt: off
    includeIDs = [234237, 324901, 759948, 25485, 324700, 43129, 266361, 108208, 40764, 130057, 39781, 49665, 39159, 23185,417218, 31251, 343072, 142076, 128100, 376725, 112500, 241530, 44563, 36950, 812276, 51018, 306013, 418105]
    # fmt: on

    def __init__(
        self,
        name: str = "Bittner-28",
        horizon: int = 11,
        reward_config: dict = None,
        end_episode_on_success: bool = False,
    ):
        graph = utils.spawn(
            file=self.genedata,
            total_genes=28,
            include_ids=self.includeIDs,
            bin_method="median",
            n_predictors=15,
            predictor_sets_path=self.predictor_sets_path,
        )

        goal_config = {
            "target_nodes": [324901],
            "intervene_on": [234237],
            "target_node_values": ((0,),),
            "undesired_node_values": tuple(),
            "horizon": horizon,
        }
        super().__init__(
            graph, goal_config, name, reward_config, end_episode_on_success
        )


class Bittner70(PBNTargetEnv):
    predictor_sets_path = Path(__file__).parent / "bittner" / "data"
    genedata = predictor_sets_path / "genedata.xls"

    includeIDs = [234237, 324901, 759948, 25485, 266361, 108208, 130057]

    N = 70
    NAME = "Bittner-70"

    def __init__(
        self,
        name: str = None,
        horizon: int = 11,
        reward_config: dict = None,
        end_episode_on_success: bool = False,
    ):
        if not name:
            name = self.NAME

        graph = utils.spawn(
            file=self.genedata,
            total_genes=self.N,
            include_ids=self.includeIDs,
            bin_method="kmeans",
            n_predictors=5,
            predictor_sets_path=self.predictor_sets_path,
        )

        goal_config = {
            "target_nodes": [324901],
            "intervene_on": [234237],
            "target_node_values": ((0,),),
            "undesired_node_values": tuple(),
            "horizon": horizon,
        }
        super().__init__(
            graph, goal_config, name, reward_config, end_episode_on_success
        )


class Bittner100(Bittner70):
    N = 100
    NAME = "Bittner-100"


class Bittner200(Bittner70):
    N = 200
    NAME = "Bittner-200"

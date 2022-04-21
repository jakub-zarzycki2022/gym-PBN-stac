from typing import List, Set, Tuple, Union

import gym
import networkx as nx
import numpy as np
from gym.spaces import Discrete, MultiBinary
from gym_PBN.types import GYM_STEP_RETURN, REWARD, STATE, TERMINATED

from .bittner import base


class PBNTargetEnv(gym.Env):
    metadata = {"render.modes": ["human", "dict", "PBN", "STG", "idx", "float"]}

    def __init__(
        self,
        graph: base.Graph,
        goal_config: dict,
        name: str = None,
        reward_config: dict = None,
    ):
        self.graph = graph

        # Goal configuration
        goal_config = self._check_config(
            goal_config,
            "goal",
            set(["target_nodes", "target_node_values", "intervene_on"]),
        )
        if goal_config is None:
            raise ValueError(
                "Target nodes, target values and intervention nodes need to be specified."
            )
        self.target_nodes = goal_config["target_nodes"]
        self.target_node_values = goal_config["target_node_values"]
        self.intervene_on = goal_config["intervene_on"]

        # Reward configuration
        reward_config = self._check_config(
            reward_config,
            "reward",
            set(["successful_reward", "wrong_attractor_cost", "action_cost"]),
            default_values={
                "successful_reward": 5,
                "wrong_attractor_cost": 2,
                "action_cost": 1,
            },
        )
        self.successful_reward = reward_config["successful_reward"]
        self.wrong_attractor_cost = reward_config["wrong_attractor_cost"]
        self.action_cost = reward_config["action_cost"]

        # Gym
        self.observation_space = MultiBinary(self.graph.N)
        self.observation_space.dtype = bool
        # intervention nodes + no action
        self.action_space = Discrete(len(self.intervene_on) + 1)
        self.name = name

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

        if action != 0:  # Action 0 is taking no action.
            _id = self.graph.getIDs().index(self.intervene_on[action - 1])
            self.graph.flipNode(_id)

        self.graph.step()

        observation = self.graph.getState()
        reward, done = self._get_reward(observation, action)
        info = {
            "observation_idx": self.render(mode="idx"),
            "observation_dict": self.render(mode="dict"),
        }

        return self.render(mode="human"), reward, done, info

    def _to_map(self, state):
        getIDs = getattr(self.graph, "getIDs", None)
        if getIDs is not None and type(state) is not dict:
            ids = [_id[0] for _id in getIDs()]
            state = dict(zip(ids, state))
        return state

    def _get_reward(self, observation: STATE, action: int) -> Tuple[REWARD, TERMINATED]:
        """The Reward function.

        Args:
            observation (STATE): The next state observed as part of the action.
            action (int): The action taken.

        Returns:
            Tuple[REWARD, TERMINATED]: Tuple of the reward and the environment done status.
        """
        reward, done = 0, False
        observation = self._to_map(observation)  # HACK Needed for some envs
        observation = tuple(
            [observation[x] for x in self.target_nodes]
        )  # Filter it down

        if observation == self.target_node_values:
            reward += self.successful_reward
            done = True
        else:
            reward -= self.wrong_attractor_cost

        if action != 0:
            reward -= self.action_cost

        return reward, done

    def reset(self):
        """Reset the environment. Initialise it to a random state, or to a certain state."""
        return self.graph.genRandState()

    def set_state(self, state: Union[List[Union[int, bool]], np.ndarray, None]):
        return self.graph.setState(state)

    def render(self, mode="human", no_cache: bool = False):
        if mode == "human":
            return list(self.graph.getState().values())
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
        else:
            raise Exception(f'Unrecognised mode "{mode}"')

    def _state_to_idx(self, state: STATE):
        return int(
            "".join([str(x) for x in list(state.values())]),
            2,
        )

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

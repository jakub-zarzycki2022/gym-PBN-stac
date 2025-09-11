import pickle
import random
from collections import defaultdict
from typing import List, Tuple
import gymnasium as gym
import networkx as nx
import numpy as np

from bang.core.pbn.utils.state_printing import convert_to_binary_representation
from gym_PBN.types import GYM_STEP_RETURN, REWARD, STATE, TERMINATED, TRUNCATED

import bang
from bang.core.attractors.monte_carlo.merge_attractors import merge_attractors


class PBNBangEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "dict", "PBN", "STG", "idx", "float", "target"]
    }

    def __init__(
        self,
        file: str,
        input_nodes: List[int],
        initial_values: list[int],
        forbidden_nodes: list[int],
        target_nodes: List[int],
        target_values: List[int],
        format: str = "sbml",
        goal_config: dict = None,
        render_mode: str = None,
        render_no_cache: bool = False,
        name: str = None,
        end_episode_on_success: bool = False,
        horizon: int = 100,
        n_parallel=2,
        pasip_history_len=10,
        pasip_size=10000,
    ):
        self.target = None
        self.end_episode_on_success = end_episode_on_success

        self.pbn = bang.load_from_file(file, format=format, n_parallel=n_parallel)
        attractors = self.pbn.monte_carlo_detect_attractors(pasip_history_len, pasip_size, repr='bool')

        self.blocks = self.pbn.get_blocks(repr='bool')
        self.all_attractors = attractors[0]
        print(attractors)

        print(f"setting {horizon}")
        self.horizon = horizon

        # Gym
        print("\nhello\n")
        self.name = name
        self.render_mode = render_mode
        self.render_no_cache = render_no_cache

        # State
        self.n_steps = 0

        self.attracting_states = set()
        self.counter = 0

        self.initial_state_id = -1
        self.target_state_id = -1

        self.target_attractor_id, self.state_attractor_id = -1, -1
        self.forbidden_actions = self.forbidden_actions = list(set(input_nodes).union(forbidden_nodes))
        self.input_nodes = input_nodes

        self.target_nodes = target_nodes
        self.target_values = target_values

        self.attractor_set = {tuple(a) for a in self.all_attractors}
        self.divided_attractors = [a for a in self.all_attractors if
                                   not (self.in_target(a))]
        self.target_attractors = [a for a in self.all_attractors if
                                  self.in_target(a)]

        if len(self.divided_attractors) == 0:
            print("THERE IS NO VALID SOURCE ATTRACTOR")
        if len(self.target_attractors) == 0:
            print("THERE IS NO VALID TARGET ATTRACTOR")

    def _seed(self, seed: int = None):
        np.random.seed(seed)
        random.seed(seed)

    def get_id(self, state):
        for i, attractor in enumerate(self.all_attractors):
            if state == attractor[0]:
                return i

        raise ValueError

    def set_input_nodes(self, input_nodes):
        self.input_nodes = input_nodes

    def step(self, actions, force=False, perturbation_prob=0.0):
        if not isinstance(actions, list):
            actions = actions.unique().tolist()

        self.n_steps += 1

        actions = [action-1 for action in actions if action not in self.forbidden_actions]

        self.pbn.simple_steps(n_steps=1, actions=actions)
        observation = self.pbn.history_bool[-1][0][0]

        while not force and not self.is_attracting_state(observation):
            self.pbn.simple_steps(n_steps=10_000)
            observation = self.pbn.history_bool[-1][-1][0]

            trajectories = self.pbn.history
            trajectories = np.squeeze(trajectories, axis=2).T
            trajectories = trajectories[::, 1:]

            attractors = merge_attractors(trajectories)
            attractors = convert_to_binary_representation(attractors, self.pbn._n)[0]

            # type II pseudo-attractor
            for a in attractors:
                if tuple(a) not in self.attracting_states:
                    self.all_attractors.append([a])
                    self.attracting_states.add(tuple(a))

                # with open(self.path, "wb+") as f:
                #     pickle.dump(self.all_attractors, f)

        reward, terminated, truncated = self._get_reward(observation, actions)

        return observation, reward, terminated, truncated, {}

    def _to_map(self, state):
        getIDs = getattr(self.graph, "getIDs", None)
        if getIDs is not None and type(state) is not dict:
            ids = getIDs()
            state = dict(zip(ids, state))
        return state

    def in_target(self, observation):
        for i in range(len(self.target_values)):
            if observation[self.target_nodes[i]] != self.target_values[i]:
                return False
        return True

    def _get_reward(self, observation: STATE, actions) -> Tuple[REWARD, TERMINATED, TRUNCATED]:

        if not isinstance(actions, list):
            actions = actions.tolist()
            actions = np.unique(actions)
        """The Reward function.

        Args:
            observation (STATE): The next state observed as part of the action.
            action (int): The action taken.

        Returns:
            Tuple[REWARD, TERMINATED, TRUNCATED]: Tuple of the reward and the environment done status.
        """
        reward, terminated = 41, False
        observation = tuple(observation)

        reward -= 1 * len(actions)

        if len(actions) > 10:
            reward -= 10 * (len(actions) - 10)

        if self.in_target(observation):
            reward += 1000
            terminated = True

        truncated = self.n_steps == self.horizon
        return reward, terminated, truncated

    def calculate_attractors(self, n_paths, path_len, trajectory_length=1000):
        if n_paths < 2:
            n_paths = 2

        self.pbn._n_parallel = n_paths
        attractors = self.pbn.monte_carlo_detect_attractors(trajectory_length=trajectory_length, attractor_length=path_len)
        return attractors

    def reset(self, seed: int = None, options: dict = None):
        """Reset the environment. Initialise it to a random state, or to a certain state."""
        if seed:
            self._seed(seed)

        state = target = None

        if len(self.divided_attractors) > 0:
            state = random.choice(self.divided_attractors)
        else:
            state = self.target_attractors[0]

        self.pbn.set_states([state])

        self.n_steps = 0
        observation = [int(x) for x in self.pbn.history_bool[-1][0][0]]
        info = {
            "observation_idx": self._state_to_idx(observation),
            "observation_di ct": observation,
        }

        self.target = None
        # print(state)
        # print(self.target_attractors)
        # print(len(self.divided_attractors))
        # print(len(self.target_attractors))
        return tuple(state), info

    def get_state(self):
        return np.array(self.graph.getState())

    def setTarget(self, target):
        self.target = target

    def render(self, mode=None):
        mode = self.render_mode if not mode else mode

        return self.pbn.history_bool[-1][0]

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
        pass

    def is_attracting_state(self, state):
        state = tuple(state)

        return state in self.attracting_states

    def get_labels(self, state):
        nodes = self.graph.nodes
        return {node.ID: s for node, s in zip(nodes, state)}

    def get_next_state(self, state, actions):
        unlabeled_state = state
        state = self.get_labels(state)
        IDs = list(state.keys())

        if not isinstance(actions, list):
            actions = actions.unique().tolist()

        for action in actions:
            if action != 0:  # Action 0 is taking no action.
                ID = IDs[action - 1]
                state[ID] = 1 - state[ID]

        i = random.randint(0, len(state) - 1)
        new_state = state
        new_state[IDs[i]] = self.graph.nodes[i].step(state)
        unlabeled_new_state = tuple(new_state.values())

        step_count = 0
        returns_count = 0
        history = defaultdict(int)
        while not self.is_attracting_state(unlabeled_new_state):  # to liczy się na jednym cpu, i prawdobodobnie powoduje bottleneck w obliczeniach
            i = random.randint(0, len(self.graph.nodes) - 1)
            new_state = state
            new_state[IDs[i]] = self.graph.nodes[i].step(state)

            if unlabeled_state == unlabeled_new_state:
                returns_count += 1
            else:
                returns_count = 0

            unlabeled_state = unlabeled_new_state
            unlabeled_new_state = tuple(new_state.values())

            state = new_state

            if returns_count > 1_000:
                print(f"append {unlabeled_state} to attractor list")
                self.all_attractors.append([unlabeled_state])
                self.attracting_states.add(unlabeled_state)
                self.probabilities.append(0)
                self.rework_probas()
                return unlabeled_state

            step_count += 1
            history[unlabeled_state] += 1

            if step_count > 10_000:
                states = sorted(history.items(), key=lambda kv: kv[1], reverse=True)
                new_attractors = [node for node, frequency in states if frequency > 1500]

                print(len(new_attractors))
                for s in new_attractors:
                    # print(s, history[s])
                    self.all_attractors.append([s])
                    self.attracting_states.add(s)
                    self.probabilities.append(0)

                self.rework_probas()
                step_count = 0

        return unlabeled_state

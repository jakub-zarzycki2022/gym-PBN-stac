# gym-PBN
An OpenAI Gym environment modelling Probabilistic Boolean Networks and Probabilistic Boolean Control Networks.

Probabilistic Boolean (Control) Networks are Boolean Networks where the logic functions for each node are switched stochastically according to a probability distribution. They were introduced by [Shmulevich, Ilya, et al., 2002](https://academic.oup.com/bioinformatics/article/18/2/261/225574?login=true) and are used primarily to model Gene Regulatory Networks. As such, their control has applications in theuraputics, and specifically cancer treatment.

The control of Probabilistic Boolean (Control) Networks is a well studied problem in control theory. Recently, however, there has been promise on the application of Reinforcement Learning for control of said networks to certain attractor states as well [Papagiannis, Georgios, et al., 2019](https://arxiv.org/abs/1909.03331).

The point of this library is to provide accessible PB(C)N environments in the OpenAI Gym framework. The environments provided are fully [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)-compatible

## Environments
- `gym_PBN:PBN-v0`: The base Probabilistic Boolean Network environment. Actions involve taking no action, or "flipping" the value of a node at the provided index.
- `gym_PBN:PBCN-v0`: The base Probabilistic Boolean Control Network environment. Actions involve setting the control nodes to a certain value.

The environments provide the framework for such networks. They need to be instantiated with appropriate node data.

## Installation
Requirements: [Python 3.6+](https://www.python.org/downloads/).

PIP: `python -m pip install gym-PBN`

Anaconda: **TO BE UPDATED**

## Configuring your own PB(C)N
Custom network environments need to be parameterised with one of the following two configurations:
1. `PBN_data`: list of tuples containing node information.
    - Each tuple should contain the following five variables, in order:
      1. `input_mask`: a boolean mask as a numpy array that indicates which nodes affect the node's value.
      2. `function`: a truth table representing a boolean function over the nodes singled out by the `input_mask`. The truth table should have a tree-like shape of `[2] * sum(input_mask)`, and the item at position `pos` should indicate the probability of the node taking the value `True` given `pos` as the state of the input nodes (which are singled out by `input_mask`.
      3. `i`: the position of the node in the network's list of vertices.
      4. `name`: string representing the name of the node. Could be `None`.
      5. `is_control`: boolean flag on whether or not this is a control node (for PBCNs).

2. `logic_func_data`: Think setting all the previous information manually is a pain? We do too, so this is the more sane configuration option. Through `logic_func_data`, you can pass in just the names of the nodes, and the associated logic functions (with their probability of activating) and the constructor will do the rest.
    - `logic_func_data` is a tuple. The tuple contains:
      1. `node_names`: a list of string literals representing each node in the network. Make sure to put control nodes at the start of the list.
      2. `logic_funcs`: a list of logic functions for each node. Each inner list (associated with the node in the corresponding position in the node names) contains tuples describing logic functions and probabilities of activating for this node. This is modelled as follows:
        1. `logic_expr`: the logic expression representing the logic function for the tuple. You can use literals that appear in the `node_names` list and Python boolean operators `and`, `not`, `or`.
        2. `probability`: a float representing the probability of this function activating.
   
   You can view an example of this second configuration over at [example.py](example.py).

#### Goal Configuration
Another thing you can configure for your own network is the actual control target. When not provided explicitly, the environment calculates the attractors for the environment and selects the last one as the target. However, especially for PBCNs, we encourage you to provide it explicitly. To do so, provide a `goal_config` argument to the environment instantiation, with the following information:

1. `"all_attractors"`: list of all the attractors present in the PB(C)N. This should be a list of sets. Each set should contain tuples that represent the attractor states if it's a cyclic attractor, or a single tuple representing the equilibruim point if it's a single-state attractor.
2. `"target"`: the target attractor, out of the list of attractors. Naturally this is a set of tuples, or just one tuple in the equilibrium point case.

#### Reward Configuration
The final thing you can configure without modifying the environment is the actual reward (and cost) values for the reward function. Simply provide a `reward_config` argument to the environment instatiation, with the following information:

1. `"successful_reward"`: integer reward given for actions that transition into the target attractor. Defaults to `5`. Recommended: `> 2`.
2. `"wrong_attractor_cost"`: integer cost associated to actions that transition into an undesired attractor. Defaults to `2`. This is applied for every attractor that the new state hits (sometimes it's multiple).
3. `"action_cost"`: integer cost associated to actions being taken. Defaults to `1`, to discourage the agent from intervening too often.


## Credits
The majority of the work for the implementation of Probabilistic Boolean Networks in Python can be attributed to [Vytenis Šliogeris](https://github.com/vjsliogeris) and his [PBN_env](https://github.com/vjsliogeris/PBN_env) package. In fact he implemented the prototype version of `gym-PBN` some time ago.

[Evangelos Chatzaroulas](mailto:e.chatzaroulas@surrey.ac.uk) finished the adaptation to the OpenAI Gym framework and implemented PB(C)N support. He is currently the primary maintainer.

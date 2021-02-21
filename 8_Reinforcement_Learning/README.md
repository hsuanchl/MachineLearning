# Rienforcement Learning

Q-learning with linear function approximation and Epsilon-greedy action selection to solve the mountain car environment.

## Usage
    python q_learning.py <mode> <weight out> <returns out> <episodes> <max iterations> <epsilon> <gamma> <learning rate>.

1. `<mode>`: mode to run the environment in. Should be either "raw" or "tile". The states are transformed into features which are one-hot encodings of the grid number in position-velocity space. The "tile" mode performs tiling/coarse coding of the state space using multiple grids offset from each other.
2. `<weight out>`: path to output the weights of the linear model
3. `<returns out>`: path to output the returns of the agent
4. `<episodes>`: the number of episodes
5. `<max iterations>`: the maximum of the length of an episode. When this is reached, we terminate the current episode.
6. `<epsilon>`: the value for the epsilon-greedy strategy
7. `<gamma>`: the discount factor γ.
8. `<learning rate>`: the learning rate α of the Q-learning algorithm

## Example
    python3 q_learning.py tile weight.out returns.out 10 200 0.05 0.99 0.01
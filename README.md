# FrozenLake using Reinforcement Learning
The objective of the project is to glide a robot agent on the frozen surface from the starting location (start state) to the frisbee (goal & terminal state) without falling into any of the 4 holes (terminal state).
The start state is at the top left location, while the frisbee is located at the bottom right corner. The operation of the robot has the following characteristics: 1) The action space of the robot has only 4 actions - left, right, up, and down. 2) The robot is confined within the grid and cannot leave the grid. Any action that moves the robot outside the grid will cause the state of the robot to remain unchanged. 3) The robot receives a reward of +1 if it reaches the frisbee, −1 if it falls into a hole, and 0 for all other cases. This reward mapping will be followed for this project. 4) An episode ends when the robot reaches any of the terminal states (reaches frisbee or falls into a hole). The instructions for task 2 are similar to task 1 except for the following rules: 1) Increase the grid size to at least 10 x 10 while maintaining the same proportion between the number of holes and states (i.e., 4/16 = 25%). 2) Distribute the holes randomly without completely blocking access to the frisbee. An example of this frozen lake grid.

In this project, I implemented tabular Reinforcement Learning methods (Q Learning, SARSA, First-Visit Monte-Carlo without Exploring Starts) to navigate an agent through any custom N by M FrozenLake grid to reach the goal.

## Install depencencies
```
pip install -r requirements.txt
```

Libraries Required:
- numpy
- tqdm
- random
- matplotlib
- PIL

## Training
Run training.ipynb notebook to train the RL algorithms.

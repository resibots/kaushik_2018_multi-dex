## Multi-DEX algorithm
Paper: [Multi-objective Model-based Policy Search for Data-efficient Learning with Sparse Rewards (CoRL 2018)](https://arxiv.org/pdf/1806.09351.pdf)

#### Abstract:
The most data-efficient algorithms for reinforcement learning in robotics are model-based policy search algorithms, which alternate between learning a dynamical model of the robot and optimizing a policy to maximize the expected return given the model and its uncertainties. However, the current algorithms lack an effective exploration strategy to deal with sparse or misleading reward scenarios: if they do not experience any state with a positive reward during the initial random exploration, it is very unlikely to solve the problem. Here, we propose a novel model-based policy search algorithm, Multi-DEX, that leverages a learned dynamical model to efficiently explore the task space and solve tasks with sparse rewards in a few episodes. To achieve this, we frame the policy search problem as a multi-objective, model-based policy optimization problem with three objectives: (1) generate maximally novel state trajectories, (2) maximize the expected return and (3) keep the system in state-space regions for which the model is as accurate as possible. We then optimize these objectives using a Pareto-based multi-objective optimization algorithm. The experiments show that Multi-DEX is able to solve sparse reward scenarios (with a simulated robotic arm) in much lower interaction time than VIME, TRPO, GEP-PG, CMA-ES and Black-DROPS.

### Dependencies
1. The algorithm and the experiments are implemented in C++-11.
2. The codes uses resouces (such as GP model learning) of limbo library. Therefore, before building the code of this repository, the limbo dependencies must be installed. [Check here](http://www.resibots.eu/limbo/tutorials/quick_start.html)
3. For Physics Simulation it uses [DART](https://dartsim.github.io/) library which must be installed in the system.
4. For random forests model learning it uses [openCV](https://opencv.org/).

The header only libraries such as sferes2(NSGA-II multi-objective optimization), limbo (GP model) and robot_dart (wrapper for DART) have been added as submodules to this repository and hence these libraries need not be installed explicitly.

### How to properly clone this repository

```bash
git clone https://github.com/resibots/kaushik_2018_multi-dex.git
git submodule init
git submodule update
```

### Configuring the experiments
```bash
cd kaushik_2018_multi-dex
./waf configure
```

### Building the experiments
```bash
./waf build -j4
```
### Running the experiments

To run without graphical visualization
```bash
./build/src/dart/sequential_goal_reaching_2dof_simu -p 0.4
./build/src/dart/drawer_opening_2dof_simu -p 0.4
```
To run with graphical visualization
```bash
./build/src/dart/sequential_goal_reaching_2dof_graphic -p 0.4
./build/src/dart/drawer_opening_2dof_graphic -p 0.4
```
Cumulative rewards observed after each execution of the policy on the robot are stored in ```results.dat``` file in the working directory.

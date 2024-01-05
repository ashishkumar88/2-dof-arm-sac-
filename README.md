### Solution of a two DOF simple arm chasing a target using Soft Actor Critic

#### Solution overview
The problem given in the playground environment is for a 2 DOF toy robotic arm to learn to reach a goal. The 2 degrees of freedom are, viz. angle of rotation of the joint attaching base with link 1 and the angle of rotation of the joint attaching the two links. 

##### Environment Modification
The original environment captures as a state a partial observation of the environment which includes state of the arms and distance of the arm tcp from the goal. The orginal environment has been modified in this work such that the state is now an image of the environment. The action space remains unchanged. The environment is then wrapped around a few wrappers provided by Gymnasium: NormalizeObservation, GrayScaleObservation, ResizeObservation to simplify the state space and finally normalize.

##### Reinforcement Learning
This environment is well suited for reinforcement learning. In this work, SAC or Soft Actor Critic is used. This algorithm was choosen because the problem is a high dimensional search problem (state, action) and an algorithm that incentivizes exploration is preferred. The other reason is my familiarity with SAC in my previous work. This work contains an implementation of a SAC trainer, a SAC algorithm implementation, implementation of the CNN critic, actor and value networks, and a logger. 

##### Directory organization
- AIM-playground - Contains the modified environment
- AIM-RL - Contains implementation of the SAC trainer and the algorithm
    - [SACTrainer](AIM-RL/trainer/sac_trainer.py) 
    - [SAC](AIM-RL/algorithm/sac.py)
    - [SACCritic](AIM-RL/model/aim_model.py)
    - [SACActor](AIM-RL/model/aim_model.py)
    - [SACValue](AIM-RL/model/aim_model.py)

#### How tos

A requirement to run the program is to set the PYTHONPATH using the following command:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/repo/AIM-RL:/path/to/repo/AIM-playground

```

##### To run the program in training mode
Before running the program in training mode, set the parameters [here](AIM-RL/config/aim_sac.yaml). After that, run the following command

```bash
python /path/to/repo/AIM-RL/trainer/sac_trainer.py --train
```

##### To run the program in evaluation mode
Firstly, set the `checkpoint_path` in the param file [here](AIM-RL/config/aim_sac.yaml). This path should point to a valid checkpoint created previously by the program. The checkpoint is located in the `base_dir`/aim_sac/`<checkpoint directory>`/checkpoints/actor.pth, where `base_dir` is also defined [here](AIM-RL/config/aim_sac.yaml). After that, run the following command

```bash
python /path/to/repo/AIM-RL/trainer/sac_trainer.py
```

#### Future work
1. Configure ML flow for experiments to deployement
2. More unit tests
3. Parallelize data collection
4. Explore attention for similar problems
5. SLAC seems nice, explore that too


#### License
This program is licensed under GPL 3, license [here](AIM-RL/gpl-3.0.txt).

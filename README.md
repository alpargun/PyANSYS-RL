# Python Interface for ANSYS and Reinforcement Learning

Defines a Gym/Gymnasium based environment for ANSYS Simulator to train reinforcement learning (RL) agents to control a linear soft pneumatic actuator (SPA) by applying air pressure to achieve a desired expansion.

# Requirements
- ANSYS license (Student is enough)
- ANSYS `.dat` solution file

# Run
- Run `ansys_test.py` to check if Python can run ANSYS with a valid license.
- Run `ansys_rl_env` to define and test the ANSYS RL environment with an agent that does random actions.
- Run `train_agent.py` to train an RL agent on the ANSYS RL environment.

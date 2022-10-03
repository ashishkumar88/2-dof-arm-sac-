#!/usr/bin/env python3

# Copyright (C) 2023 Ashish Kumar
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program in the file: gpl-3.0.text. 
# If not, see <http://www.gnu.org/licenses/>.

# system imports
import argparse
import datetime
import os
import random

# library imports
import torch
import numpy as np
from gymnasium.wrappers import ResizeObservation, GrayScaleObservation, NormalizeObservation
from torchrl.data import ReplayBuffer, LazyMemmapStorage
from tensordict.tensordict import TensorDict

# local imports
from env import ArmEnv
from config.loader import load_config
from algorithm.sac import SAC
from logger.logger import create_logger

logger = create_logger("aim_sac_trainer")

class SACTrainer:
    """ SAC Trainer implementation.

        Reference: https://spinningup.openai.com/en/latest/algorithms/sac.html
        The trainer first initializes all the training parameters from the config file. The trainer
        then initializes the SAC algorithm and the environment. The trainer also initializes the replay
        buffer and fills it with random actions. The trainer then starts the training loop. In each iteration,
        the trainer samples a batch of data from the replay buffer and performs the updates. The trainer also
        logs the training progress to tensorboard and the terminal. The trainer also saves the model checkpoints
        to the checkpoints directory.
    """
    def __init__(self, config_path=None) -> None:
        
        if config_path is None or os.path.isfile(config_path) is False:
            config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../config/aim_sac.yaml")

        trainer_config = load_config(config_path)

        # SAC Trainer Configs
        for config in trainer_config:
            setattr(self, config, trainer_config[config])   

        print("SACTrainer Configs:")
        for attr_name in dir(self):
            if not attr_name.startswith("__") and not callable(getattr(self, attr_name)):
                print(f"{attr_name}: {getattr(self, attr_name)}")

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

        if self.base_dir is None:
            self.base_dir = os.path.join(os.environ["HOME"], "aim-rl")

        self.base_dir = os.path.join(self.base_dir, self.experiment_name, timestamp)
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            os.makedirs(os.path.join(self.base_dir, "checkpoints"))

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._total_steps = 0

        self._env = NormalizeObservation(GrayScaleObservation(ResizeObservation(ArmEnv(), 84), keep_dim=True))
        self._sac = SAC(self._env, self.base_dir, self.gamma, self.actor_lr, self.critic_lr, self.alpha_lr, self.tau, self.initial_alpha, self.target_entropy, self.batch_size, self.replay_buffer_size, self._device, self.number_updates_per_iter, self.target_network_update_rate)
        self._replay_buffer = ReplayBuffer(storage=LazyMemmapStorage(max_size=self.replay_buffer_size), batch_size=self.batch_size,)

    def _change_observation_to_tensor(self, observation: np.ndarray) -> torch.Tensor:
        observation = np.expand_dims(observation, axis=0)
        observation = torch.from_numpy(observation.copy()).float().to(self._device)
        observation = observation.permute(0, 3, 1, 2)
        return observation
    
    def _change_action_to_tensor(self, action: np.ndarray) -> torch.Tensor:
        action = np.expand_dims(action, axis=0)
        action = torch.from_numpy(action).float().to(self._device)
        assert(action.shape == (1, 2))
        return action
    
    def train(self) -> None:
        """ Training loop for SAC as described here: https://spinningup.openai.com/en/latest/algorithms/sac.html
        """
        observation = self._reset_env()

        for _ in range(self.initialization_steps):
            action = self._env.action_space.sample()
            observation_new, reward, done, truncate, info = self._env.step(action)
            observation_new = self._change_observation_to_tensor(observation_new)
            action = self._change_action_to_tensor(action)

            # write to replay buffer
            sars = TensorDict({
                "observation": observation,
                "action": action,
                ("next", "reward"): torch.tensor([reward]).to(torch.float32).unsqueeze(0).to(self._device),
                ("next", "observation"): observation_new,
                ("next", "done"): torch.tensor([done]).to(torch.bool).unsqueeze(0).to(self._device),
                ("next", "terminated"): torch.tensor([False]).to(torch.bool).unsqueeze(0).to(self._device)
            }, 1, device=self._device)
            self._replay_buffer.extend(sars)

            observation = observation_new

            if done:
                observation, _ = self._env.reset()
                observation = self._change_observation_to_tensor(observation)

        for _ in range(self.timesteps_total):
            self._total_steps += 1

            action = self._sac.generate_action(TensorDict({"observation": observation}, 1, device=self._device))
            observation_new, reward, done, truncate, info = self._env.step(action)
            observation_new = self._change_observation_to_tensor(observation_new)
            action = self._change_action_to_tensor(action)

            # write to replay buffer
            sars = TensorDict({
                "observation": observation,
                "action": action,
                ("next", "reward"): torch.tensor([reward]).to(torch.float32).unsqueeze(0).to(self._device),
                ("next", "observation"): observation_new,
                ("next", "done"): torch.tensor([done]).to(torch.bool).unsqueeze(0).to(self._device),
                ("next", "terminated"): torch.tensor([False]).to(torch.bool).unsqueeze(0).to(self._device)
            }, 1, device=self._device) 
            self._replay_buffer.extend(sars)
            observation = observation_new

            if self._total_steps % self.number_steps_per_iter == 0:
                log = self._sac.perform_updates(self._replay_buffer, self._total_steps)

                # write to terminal
                logger.info(f"Performed update. Total Steps: {self._total_steps}.")
                logger.info(f"Loss Actor: {log['loss_actor']}.")
                logger.info(f"Loss Alpha: {log['loss_alpha']}.")
                logger.info(f"Loss QValue: {log['loss_qvalue']}.")
                logger.info(f"Loss Value: {log['loss_value']}.")
                logger.info(f"Mean Reward: {log['reward']}.")

                if log["reward"] is not None and log["reward"] >= self.stop_mean_reward:
                    logger.info("Stopping training.")
                    break
                observation = self._reset_env()

            if done:
                observation = self._reset_env()

    def _reset_env(self) -> None:
        observation, _ = self._env.reset(seed=random.randint(10, 1e6))
        observation = self._change_observation_to_tensor(observation)
        return observation

    def eval(self) -> None:
        if os.path.exists(self.checkpoint_path) is False or os.path.isfile(self.checkpoint_path) is False:
            logger.error(f"Checkpoint path {self.checkpoint_path} does not exist. Exiting.")
            return
        
        self._sac.load_model(self.checkpoint_path)

        observation, _ = self._env.reset(seed=random.randint(10, 1e6))
        observation = self._change_observation_to_tensor(observation)

        for _ in range(self.evaluation_steps):
            self._total_steps += 1

            action = self._sac.generate_action(observation, eval=True)
            observation, reward, done, truncate, info = self._env.step(action)
            observation = self._change_observation_to_tensor(observation)

            if done:                
                observation, _ = self._env.reset(seed=random.randint(10, 1e6))
                observation = self._change_observation_to_tensor(observation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train or evaluate', default=False)

    args = parser.parse_args()
    
    trainer = SACTrainer()

    if args.train:
        logger.info("Strating in training mode.")
        trainer.train()
    else:
        logger.info("Strating in evaluation mode.")
        trainer.eval()
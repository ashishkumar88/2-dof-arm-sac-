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
import os
import sys

# external imports
import ray
from ray import air, tune
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.registry import register_env
from gymnasium.wrappers import ResizeObservation
import torch

# internal imports
from config.loader import load_config
from env import ArmEnv

class SACTrainer:

    def __init__(self, config_path=None):
        
        if config_path is None or os.path.isfile(config_path) is False:
            config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../config/aim_sac.yaml")

        trainer_config = load_config(config_path)

        # SAC Trainer Configs
        self._gamma = trainer_config["gamma"] if "gamma" in trainer_config else 0.99
        self._actor_lr = trainer_config["actor_lr"] if "actor_lr" in trainer_config else 0.0003
        self._critic_lr = trainer_config["critic_lr"] if "critic_lr" in trainer_config else 0.0003
        self._entropy_lr = trainer_config["entropy_lr"] if "entropy_lr" in trainer_config else 0.0003
        self._tau = trainer_config["tau"] if "tau" in trainer_config else 0.005
        self._initial_alpha = trainer_config["initial_alpha"] if "initial_alpha" in trainer_config else 0.2
        self._target_entropy = trainer_config["target_entropy"] if "target_entropy" in trainer_config else "auto"
        self._n_step = trainer_config["n_step"] if "n_step" in trainer_config else 1
        self._use_gpu = trainer_config["use_gpu"] if "use_gpu" in trainer_config else False
        self._device = torch.device("cuda" if torch.cuda.is_available() and self._use_gpu else "cpu")
        self._num_gpus = trainer_config["num_gpus"] if "num_gpus" in trainer_config else 1
        self._num_cpus = trainer_config["num_cpus"] if "num_cpus" in trainer_config else 0
        self._base_dir = trainer_config["base_dir"] if "base_dir" in trainer_config else os.path.join(os.environ["HOME"], "aim-rl")
        self._train_batch_size = trainer_config["train_batch_size"] if "train_batch_size" in trainer_config else 256
        self._target_network_update_freq = trainer_config["target_network_update_freq"] if "target_network_update_freq" in trainer_config else 32
        self._timesteps_total = trainer_config["timesteps_total"] if "timesteps_total" in trainer_config else 10000000
        self._num_workers = trainer_config["num_workers"] if "num_workers" in trainer_config else 1
        self._buffer_size = trainer_config["buffer_size"] if "buffer_size" in trainer_config else 1000000
        
        # Print class attributes with their names
        print("SACTrainer Configs:")
        for attr_name in dir(self):
            if not attr_name.startswith("__") and not callable(getattr(self, attr_name)):
                print(f"{attr_name}: {getattr(self, attr_name)}")

        # Environment Configs
        self._env = ResizeObservation(ArmEnv(), 84)
        register_env("arm_env", lambda _: self._env)

        # SACConfigs
        if os.path.isdir(self._base_dir) is False:
            os.makedirs(self._base_dir)

        self._sac_config = SACConfig()\
            .framework(framework="torch")\
            .environment(env='arm_env', render_env=False,)
        
        self._sac_config.min_sample_timesteps_per_iteration = self._train_batch_size

        if self._use_gpu:
            self._sac_config = self._sac_config.resources(num_gpus=self._num_gpus)
        else:
            self._sac_config = self._sac_config.resources(num_cpus=self._num_cpus)

        self._sac_config = self._sac_config.training(
            gamma=self._gamma,
            target_network_update_freq=self._target_network_update_freq,
            tau=self._tau,
            train_batch_size=self._train_batch_size,
            replay_buffer_config={'capacity': self._buffer_size},
            initial_alpha=self._initial_alpha,
            n_step=self._n_step,
            target_entropy=self._target_entropy,
            optimization_config={'actor_learning_rate': self._actor_lr, 'critic_learning_rate': self._critic_lr, 'entropy_learning_rate': self._entropy_lr},
        )

        self._air_config = air.RunConfig(
            name='SAC',
            stop={'timesteps_total': self._timesteps_total},
            checkpoint_config=air.CheckpointConfig(
                checkpoint_at_end=True
            ),
            local_dir=self._base_dir,
            log_to_file=True,
        )

        self._tuner = tune.Tuner(
            'SAC',
            param_space=self._sac_config,
            run_config=self._air_config,
        )        

    def train(self):
        ray.init()
        self._tuner.fit()
        ray.shutdown()

    def eval(self):
        pass


if __name__ == "__main__":
    trainer = SACTrainer()
    trainer.train()
    trainer.eval()
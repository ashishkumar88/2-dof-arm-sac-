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
from typing import Dict
import os

# library imports
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor
from torchrl.data import ReplayBuffer
from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
from torchrl.data import BoundedTensorSpec
from torchrl.modules.distributions.continuous import TanhNormal, NormalParamWrapper
from tensordict.nn import TensorDictModule, InteractionType
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.objectives.sac import SACLoss
from torchrl.objectives import SoftUpdate
from torchrl.modules.tensordict_module.common import SafeModule

# local imports
from model.aim_model import SACCritic, SACActor, SACValue
from logger.logger import create_logger

logger = create_logger("aim_sac")

class SAC:
    """ SAC algorithm implementation.

        The implementation is based on this tutorial: https://pytorch.org/rl/reference/generated/torchrl.objectives.SACLoss.html
    """
    def __init__(self,
            env,
            base_dir: str,
            gamma: float = 0.99,
            actor_lr: float = 3e-4,
            critic_lr: float = 3e-4,
            alpha_lr: float = 3e-4,
            tau: float = 0.005,
            initial_alpha: float = 0.1,
            target_entropy: float = "auto",
            batch_size: int = 256,
            replay_buffer_size: int = 1000000,
            device = torch.device("cuda:0"),
            number_updates_per_iter: int = 100,
            target_network_update_rate: float = 0.995,
            value_lr: float = 3e-4,
        ) -> None:
        
        self._env = env
        self._gamma = gamma
        self._actor_lr = actor_lr
        self._critic_lr = critic_lr
        self._tau = tau
        self._initial_alpha = initial_alpha
        self._target_entropy = target_entropy
        self._batch_size = batch_size
        self._replay_buffer_size = replay_buffer_size
        self._device = device
        self._number_updates_per_iter = number_updates_per_iter
        self._base_dir = base_dir
        self._alpha_lr = alpha_lr
        self._value_lr = value_lr
        self._target_network_update_rate = target_network_update_rate
        
        self._summary_writer = SummaryWriter(os.path.join(base_dir, "tb_logs"))

        # sac modules
        n_act = self._env.action_space.shape[0] 
        spec = BoundedTensorSpec(-torch.ones(n_act), torch.ones(n_act), (n_act,))
        
        self._actor_net = SACActor(num_in_channels=self._env.observation_space.shape[2], action_dims=n_act, device=device)
        
        actor_module = NormalParamWrapper(self._actor_net)
        actor_module = SafeModule(
            actor_module, 
            in_keys=["observation"], 
            out_keys=["loc", "scale"]
        )

        self._actor = ProbabilisticActor(
            module=actor_module, 
            in_keys=["loc", "scale"], 
            spec=spec, 
            distribution_class=TanhNormal,
            default_interaction_type=InteractionType.RANDOM,
        )

        self._q_value_module = SACCritic(num_in_channels=self._env.observation_space.shape[2], device=self._device)
        self._q_value = ValueOperator(module=self._q_value_module, in_keys=['observation', 'action'])

        self._value_module = SACValue(num_in_channels=self._env.observation_space.shape[2], device=device)
        self._value = ValueOperator(module=self._value_module, in_keys=["observation"])

        self._sac_loss = SACLoss(
            self._actor, 
            self._q_value, 
            self._value,
            target_entropy=self._target_entropy,
            alpha_init=self._initial_alpha,
            gamma=self._gamma,
            loss_function='l2')

        self._sac_loss.make_value_estimator(gamma=self._gamma)
        self._target_net_updater = SoftUpdate(self._sac_loss, eps=self._target_network_update_rate)

        # sac optimizers
        critic_params = list(self._sac_loss.qvalue_network_params.flatten_keys().values())
        actor_params = list(self._sac_loss.actor_network_params.flatten_keys().values())
        value_params = list(self._sac_loss.value_network_params.flatten_keys().values())

        self._actor_optimizer = Adam(
            actor_params,
            lr=self._actor_lr,
            weight_decay=5e-5,
            eps=1e-6,
        )

        self._critic_optimizer = Adam(
            critic_params,
            lr=self._critic_lr,
            weight_decay=5e-5,
            eps=1e-6,
        )

        self._value_optimizer = Adam(
            value_params,
            lr=self._value_lr,
            weight_decay=5e-5,
            eps=1e-6,
        )

        self._alpha_optimizer = Adam(
            [self._sac_loss.log_alpha],
            lr=self._alpha_lr,
        )

    def generate_action(self, observation: Tensor, eval=False) -> Dict:
        if eval:
            action = self._actor_net(observation)
            action = action[0].detach().cpu().numpy()[0:2]
            return action
        else:
            action = self._actor(observation)
            action = action["action"].detach().cpu().numpy()[0]
            return action
    
    def perform_updates(self, replay_buffer: ReplayBuffer, total_steps: int):
        loss_actor = []
        loss_alpha = []
        loss_qvalue = []
        loss_value = []
        reward = []

        for _ in range(self._number_updates_per_iter):
            sample = replay_buffer.sample(self._batch_size) 
            sample = sample.to(self._device)
            loss = self._sac_loss(sample)

            self._actor_optimizer.zero_grad()
            loss["loss_actor"].backward()
            self._actor_optimizer.step()

            self._alpha_optimizer.zero_grad()
            loss["loss_alpha"].backward()
            self._alpha_optimizer.step()

            self._critic_optimizer.zero_grad()
            loss["loss_qvalue"].backward()
            self._critic_optimizer.step()

            self._value_optimizer.zero_grad()
            loss["loss_value"].backward()
            self._value_optimizer.step()

            self._target_net_updater.step()

            loss_actor.append(loss["loss_actor"].item())
            loss_alpha.append(loss["loss_alpha"].item())
            loss_qvalue.append(loss["loss_qvalue"].item())
            loss_value.append(loss["loss_value"].item())
            reward.append(sample["next"]["reward"].mean().item())

        # write tensorboard logs
        log = {
                'loss_actor' : sum(loss_actor) / len(loss_actor),
                'loss_alpha' : sum(loss_alpha) / len(loss_alpha),
                'loss_qvalue' : sum(loss_qvalue) / len(loss_qvalue),
                'loss_value' : sum(loss_value) / len(loss_value),
                'reward': sum(reward) / len(reward),
            }
        
        self._summary_writer.add_scalars(
            "sac",
            log,
            total_steps
        )

        # save model
        torch.save({
            'actor_state_dict': self._actor_net.state_dict(), # TODO: Add critic and value state dict
            }, os.path.join(self._base_dir, "checkpoints", "actor.pth"))

        return log

    def load_model(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise ValueError("Checkpoint path does not exist.")
        
        if not os.path.isfile(checkpoint_path):
            raise ValueError("Checkpoint path is not a file.")
        
        if checkpoint_path.endswith(".pth") is False:
            raise ValueError("Checkpoint path does not end with .pth")

        checkpoint = torch.load(checkpoint_path)
        self._actor_net.load_state_dict(checkpoint['actor_state_dict'])
        self._actor_net.eval()

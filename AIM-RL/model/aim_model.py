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

# library imports
import torch
from torch import nn
import numpy as np

class SACCritic(torch.nn.Module):
    def __init__(self, 
            num_in_channels: int = 1,
            conv_1_out_channels: int = 32,
            conv_2_out_channels: int = 64,
            conv_1_kernel_size: tuple = (7, 7),
            conv_2_kernel_size: tuple = (7, 7),
            conv_3_kernel_size: tuple = (5, 5),
            hidden_size = 256,
            action_dims = 2,
            device = torch.device("cuda:0"),
        ) -> None:
        super().__init__()
        self._device = device

        # cnn block # TODO: create a base mixin class for cnn blocks
        self._conv_1 = nn.Conv2d(in_channels=num_in_channels, out_channels=conv_1_out_channels, kernel_size=conv_1_kernel_size, padding=0, stride=2, device=self._device)
        self._elu_1 = nn.ELU(alpha=1.0)
        self._conv_2 = nn.Conv2d(in_channels=conv_1_out_channels, out_channels=conv_2_out_channels, kernel_size=conv_2_kernel_size, padding=0, stride=2, device=self._device)
        self._elu_2 = nn.ELU(alpha=1.0)

        self._conv_3 = nn.Conv2d(in_channels=conv_2_out_channels, out_channels=(hidden_size), kernel_size=conv_3_kernel_size, padding=0, stride=1, device=self._device)
        self._conv_3_flatten = nn.Flatten()
        self._elu_3 = nn.ELU(alpha=1.0)

        # fcc block
        self._fc_1 = nn.Linear(43264 + action_dims, hidden_size, device=self._device) # 43264 is the output of the cnn block, TODO: make this dynamic
        self._dropout_1 = nn.Dropout(p=1.0)
        self._fc_2 = nn.Linear(hidden_size, hidden_size, device=self._device)
        self._dropout_2 = nn.Dropout(p=1.0)
        self._elu_4 = nn.ELU(alpha=1.0)

        self._critic_linear = nn.Linear(hidden_size, 1, device=self._device)
        self.to(self._device)
        self.train()
    
    def forward(self, state, action):
        assert state.shape[2] == 84 and state.shape[3] == 84, "Currently only supports 84x84 images"
        state = self._conv_1(state)
        state = self._elu_1(state)
        state = self._conv_2(state)
        state = self._elu_2(state)
        state = self._conv_3(state)
        state = self._conv_3_flatten(state)
        state = self._elu_3(state)

        state_action = torch.cat([state, action], 1).to(self._device)
        state_action = self._fc_1(state_action)
        state_action = self._dropout_1(state_action)
        state_action = self._fc_2(state_action)
        state_action = self._dropout_2(state_action)
        state_action = self._elu_4(state_action)

        return self._critic_linear(state_action)


class SACActor(torch.nn.Module):
    def __init__(self, 
            num_in_channels: int = 1,
            conv_1_out_channels: int = 32,
            conv_2_out_channels: int = 64,
            conv_1_kernel_size: tuple = (7, 7),
            conv_2_kernel_size: tuple = (7, 7),
            conv_3_kernel_size: tuple = (5, 5),
            hidden_size = 256,
            action_dims = 2,
            device = torch.device("cuda:0"),
        ) -> None:
        super().__init__()
        self._device = device

        # cnn block # TODO: create a base mixin class for cnn blocks
        self._conv_1 = nn.Conv2d(in_channels=num_in_channels, out_channels=conv_1_out_channels, kernel_size=conv_1_kernel_size, padding=0, stride=2, device=self._device)
        self._elu_1 = nn.ELU(alpha=1.0)
        self._conv_2 = nn.Conv2d(in_channels=conv_1_out_channels, out_channels=conv_2_out_channels, kernel_size=conv_2_kernel_size, padding=0, stride=2, device=self._device)
        self._elu_2 = nn.ELU(alpha=1.0)

        self._conv_3 = nn.Conv2d(in_channels=conv_2_out_channels, out_channels=(hidden_size), kernel_size=conv_3_kernel_size, padding=0, stride=1, device=self._device)
        self._conv_3_flatten = nn.Flatten()
        self._elu_3 = nn.ELU(alpha=1.0)

        # fcc block
        self._fc_1 = nn.Linear(43264, hidden_size, device=self._device) # 43264 is the output of the cnn block, TODO: make this dynamic
        self._dropout_1 = nn.Dropout(p=1.0)
        self._fc_2 = nn.Linear(hidden_size, hidden_size, device=self._device)
        self._dropout_2 = nn.Dropout(p=1.0)
        self._elu_4 = nn.ELU(alpha=1.0)

        self._actor_linear = nn.Linear(hidden_size, 2*action_dims, device=self._device)

        self.to(self._device)
        self.train()
    
    def forward(self, state):
        assert state.shape[2] == 84 and state.shape[3] == 84, "Currently only supports 84x84 images"
        state = self._conv_1(state)
        state = self._elu_1(state)
        state = self._conv_2(state)
        state = self._elu_2(state)
        state = self._conv_3(state)
        state = self._conv_3_flatten(state)
        state = self._elu_3(state)

        state = self._fc_1(state)
        state = self._dropout_1(state)
        state = self._fc_2(state)
        state = self._dropout_2(state)
        state = self._elu_4(state)

        return self._actor_linear(state)


class SACValue(torch.nn.Module):
    def __init__(self, 
            num_in_channels: int = 1,
            conv_1_out_channels: int = 32,
            conv_2_out_channels: int = 64,
            conv_1_kernel_size: tuple = (7, 7),
            conv_2_kernel_size: tuple = (7, 7),
            conv_3_kernel_size: tuple = (5, 5),
            hidden_size = 256,
            device = torch.device("cuda:0"),
        ) -> None:
        super().__init__()
        self._device = device

        # cnn block # TODO: create a base mixin class for cnn blocks
        self._conv_1 = nn.Conv2d(in_channels=num_in_channels, out_channels=conv_1_out_channels, kernel_size=conv_1_kernel_size, padding=0, stride=2, device=self._device)
        self._elu_1 = nn.ELU(alpha=1.0)
        self._conv_2 = nn.Conv2d(in_channels=conv_1_out_channels, out_channels=conv_2_out_channels, kernel_size=conv_2_kernel_size, padding=0, stride=2, device=self._device)
        self._elu_2 = nn.ELU(alpha=1.0)

        self._conv_3 = nn.Conv2d(in_channels=conv_2_out_channels, out_channels=(hidden_size), kernel_size=conv_3_kernel_size, padding=0, stride=1, device=self._device)
        self._conv_3_flatten = nn.Flatten()
        self._elu_3 = nn.ELU(alpha=1.0)

        # fcc block
        self._fc_1 = nn.Linear(43264, hidden_size, device=self._device) # 43264 is the output of the cnn block, TODO: make this dynamic
        self._dropout_1 = nn.Dropout(p=1.0)
        self._fc_2 = nn.Linear(hidden_size, hidden_size, device=self._device)
        self._dropout_2 = nn.Dropout(p=1.0)
        self._elu_4 = nn.ELU(alpha=1.0)

        self._value_linear = nn.Linear(hidden_size, 1, device=self._device)

        self.to(self._device)
        self.train()
    
    def forward(self, state):
        assert state.shape[2] == 84 and state.shape[3] == 84, "Currently only supports 84x84 images"
        state = self._conv_1(state)
        state = self._elu_1(state)
        state = self._conv_2(state)
        state = self._elu_2(state)
        state = self._conv_3(state)
        state = self._conv_3_flatten(state)
        state = self._elu_3(state)

        state = self._fc_1(state)
        state = self._dropout_1(state)
        state = self._fc_2(state)
        state = self._dropout_2(state)
        state = self._elu_4(state)

        return self._value_linear(state)
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
import pytest
import torch
import numpy as np
from gymnasium.wrappers import ResizeObservation

# local imports
from env import ArmEnv
from model.aim_model import SACCritic, SACActor, SACValue

@pytest.fixture(scope="session")
def env():
    env = ResizeObservation(ArmEnv(), 84)
    yield env
    env.close()

@pytest.fixture(scope="session")
def critic(env):
    observation_space = env.observation_space.shape
    action_space = env.action_space.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    critic = SACCritic(num_in_channels=observation_space[2], action_dims=action_space[0], device=device)
    yield critic

@pytest.fixture(scope="session")
def actor(env):
    observation_space = env.observation_space.shape
    action_space = env.action_space.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    actor = SACActor(num_in_channels=observation_space[2], action_dims=action_space[0], device=device)
    yield actor

@pytest.fixture(scope="session")
def value(env):
    observation_space = env.observation_space.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    value = SACValue(num_in_channels=observation_space[2], device=device)
    yield value

def test_critic(env, critic):
    for _ in range(1000):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        observation = env.reset()
        action = env.action_space.sample()
        observation, reward, done, truncate, info = env.step(action)
        observation = np.expand_dims(observation, axis=0)
        observation = torch.from_numpy(observation.copy()).float().to(device)
        observation = observation.permute(0, 3, 1, 2)
        action = np.expand_dims(action, axis=0)
        action = torch.from_numpy(action).float().to(device)
        assert(critic.forward(observation, action).shape == (1, 1))

def test_actor(env, actor):
    for _ in range(1000):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        observation = env.reset()
        action = env.action_space.sample()
        observation, reward, done, truncate, info = env.step(action)
        observation = np.expand_dims(observation, axis=0)
        observation = torch.from_numpy(observation.copy()).float().to(device)
        observation = observation.permute(0, 3, 1, 2)
        assert(actor.forward(observation).shape == (1, 4))

def test_value(env, value):
    for _ in range(1000):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        observation = env.reset()
        action = env.action_space.sample()
        observation, reward, done, truncate, info = env.step(action)
        observation = np.expand_dims(observation, axis=0)
        observation = torch.from_numpy(observation.copy()).float().to(device)
        observation = observation.permute(0, 3, 1, 2)
        assert(value.forward(observation).shape == (1, 1))
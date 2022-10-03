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

import os 

import yaml

def load_config(file_path: str):
    """Load library configuration from a yaml file
    """
    if os.path.exists(file_path) and os.path.isfile(file_path) and file_path.endswith("yaml"):
        config = yaml.safe_load(open(file_path))
        return config
    else:
        raise ValueError("The config file must be a valid yaml file.")
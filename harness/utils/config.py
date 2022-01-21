# Augur: A Step Towards Realistic Drift Detection in Production MLSystems - Code
# Copyright 2022 Carnegie Mellon University.
# 
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
# 
# Released under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
# 
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.
# 
# Carnegie Mellon® is registered in the U.S. Patent and Trademark Office by Carnegie Mellon University.
# 
# This Software includes and/or makes use of the following Third-Party Software subject to its own license:
# 1. Tensorflow (https://github.com/tensorflow/tensorflow/blob/master/LICENSE) Copyright 2014 The Regents of the University of California.
# 2. Pandas (https://github.com/pandas-dev/pandas/blob/main/LICENSE) Copyright 2021 AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team, and open source contributors.
# 3. scikit-learn (https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) Copyright 2021 The scikit-learn developers.
# 4. numpy (https://github.com/numpy/numpy/blob/main/LICENSE.txt) Copyright 2021 NumPy Developers.
# 5. scipy (https://github.com/scipy/scipy/blob/main/LICENSE.txt) Copyright 2021 SciPy Developers.
# 6. statsmodels (https://github.com/statsmodels/statsmodels/blob/main/LICENSE.txt) Copyright 2018 Jonathan E. Taylor, Scipy developers, statsmodels Developers.
# 7. matplotlib (https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE) Copyright 2016 Matplotlib development team.
# 
# DM22-0044

import os
import json


class Config(object):
    """ Handles a JSON configuration."""
    config_data = {}
    config_filename = ""

    def load(self, config_filename):
        """ Creates a parser for the default config file, if it wasn't loaded before."""
        with open(config_filename) as config_file:
            self.config_data = json.load(config_file)
        self.config_filename = config_filename

    def get(self, key_name):
        """Returns a dict with the default values.#"""
        if key_name in self.config_data:
            return self.config_data[key_name]
        else:
            raise Exception("Config key not found: " + key_name)

    def contains(self, key_name):
        """Checks whether a specific config is there."""
        return key_name in self.config_data

    @staticmethod
    def get_config_file(arguments, config_folder, default_config):
        """"Gets the config file to use, either deafult, argument, from exp folder, or from user selection."""
        if arguments.config:
            config_file = arguments.config
            print(f"Using config passed as argument: {config_file}")
        elif arguments.exp_config:
            config_file = os.path.join(config_folder, arguments.exp_config)
            print(f"Using config file from experiments folder: {config_file}")
        elif arguments.exp_user:
            config_file = Config.get_file_option_from_user(config_folder)
        else:
            print("Using default config: " + default_config)
            config_file = default_config

        return config_file

    @staticmethod
    def get_file_option_from_user(config_folder):
        print("Please select one of the configuration files from the list:")
        file_idx = 0
        config_files = []
        for root, dirs, files in os.walk(config_folder):
            for filename in files:
                file_idx = file_idx + 1
                config_files.append(filename)
                print(f"{file_idx}. {filename}")
        try:
            option = int(input(f"Which file would you use? (Indicate a number between 1 and {file_idx}): "))
        except ValueError as ex:
            print("Invalid option selected.")
            return None

        if option not in range(1, file_idx+1):
            print("No valid config file selected.")
            return None
        else:
            config_file = os.path.join(config_folder, config_files[option-1])
            print("Config file to use: ", config_file)
            return config_file

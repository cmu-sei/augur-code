import os
import json


class Config(object):
    """ Handles a JSON configuration."""
    config_data = {}

    def load(self, config_filename):
        """ Creates a parser for the default config file, if it wasn't loaded before."""
        with open(config_filename) as config_file:
            self.config_data = json.load(config_file)

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

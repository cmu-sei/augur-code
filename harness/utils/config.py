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
    def choose_from_folder(arguments, config_folder, default_config):
        """Processes command line arguments to allow for 1) choosing a config from a given folder, or
        2) to pass a config file name as a parameter."""
        if len(arguments) > 1:
            command = str(arguments[1])
            if command == "--exp":
                print("Please select one of the configuration files from the list:")
                file_idx = 0
                config_files = []
                for root, dirs, files in os.walk(config_folder):
                    for filename in files:
                        file_idx = file_idx + 1
                        config_files.append(filename)
                        print(f"{file_idx}. {filename}")
                try:
                    option = int(input("Which file would you use? (Indicate a number): "))
                    if option not in range(1, file_idx):
                        print("No valid config file selected.")
                    else:
                        config_file = os.path.join(config_folder, config_files[option-1])
                        print("Config file to use: ", config_file)
                        return config_file
                except ValueError as ex:
                    # Assume we are getting a config file name.
                    config_file = command
                    print(f"Config file to use: {config_file}")
                    return config_file
            else:
                print("No valid command line arguments found.")
        else:
            print("No command line arguments.")

        print("Using default config: " + default_config)
        return default_config

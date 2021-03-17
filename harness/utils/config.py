import json


# Handles a JSON configuration.
class Config(object):
    config_data = {}

    # Creates a parser for the default config file, if it wasn't loaded before.
    def load(self, config_filename):
        with open(config_filename) as config_file:
            self.config_data = json.load(config_file)

    # Returns a dict with the default values.
    def get(self, key_name):
        if key_name in self.config_data:
            return self.config_data[key_name]
        else:
            raise Exception("Config key not found: " + key_name)

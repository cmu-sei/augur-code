
from utils import dataframe_helper
from utils.config import Config

CONFIG_FILENAME = "./merger_config.json"


# Main code.
def main():
    # Load config.
    config = Config()
    config.load(CONFIG_FILENAME)

    dataframe_helper.merge_files(config.get("dataset1"), config.get("dataset2"), config.get("output"))


if __name__ == '__main__':
    main()

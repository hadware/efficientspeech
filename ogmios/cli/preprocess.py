from pathlib import Path

import yaml
from tap import Tap

from ogmios.dataset.preprocessor import Preprocessor


class PreprocessCommand(Tap):
    config: Path  # Path to config file (yaml)


if __name__ == "__main__":
    args = PreprocessCommand().parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()

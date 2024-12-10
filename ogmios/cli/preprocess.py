import logging
from pathlib import Path

import yaml
from tap import Tap

from ogmios.dataset.commons import PreprocessingConfig, DatasetFolder
from ogmios.dataset.preprocessor import DatasetPreprocessor
from ogmios.utils import logger


class PreprocessCommand(Tap):
    verbose: bool = False
    config: Path  # Path to config file (yaml)


if __name__ == "__main__":
    args = PreprocessCommand().parse_args()
    logger.setLevel(level=logging.DEBUG if args.verbose else logging.INFO)

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    preprocessing_config = PreprocessingConfig(**config["preprocessing"])
    dataset_config = DatasetFolder(root_path=Path(config["dataset"]["root_path"]),
                                   ds_name=config["dataset"].get("name"))
    preprocessor = DatasetPreprocessor(preprocessing_config, dataset_config)
    preprocessor.process_files()
    preprocessor.save_splits()
    preprocessor.save_stats()
    preprocessor.save_phones()
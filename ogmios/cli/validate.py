from pathlib import Path

from tap import Tap


class ValidateCommand(Tap):
    dataset_root: Path  # Path to dataset root


if __name__ == "__main__":
    args = ValidateCommand().parse_args()

    # TODO: check if required folders are here, and that all files are properly referenced



from pathlib import Path

from .models import Generator

HIFIGAN_MODELS_FOLDER = Path(__name__).parent

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
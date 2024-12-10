
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator


class Phonemizer:

    def __init__(self, lang: str):
        backend = EspeakBackend(lang)
        separator = Separator(phone='|', word="")

    def __call__(self, text: str):
        return


        # build the lexicon by phonemizing each word one by one. The backend.phonemize
        # function expect a list as input and outputs a list.
        lexicon = {
            word: backend.phonemize([word], separator=separator, strip=True)[0]
            for word in words}
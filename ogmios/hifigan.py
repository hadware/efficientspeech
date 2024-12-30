from pathlib import Path

import numpy as np
from onnxruntime import InferenceSession


class HifiganOnnxModel:

    def __init__(self, path: Path):
        self.onnx_session = InferenceSession(str(path))

    def synth(self, mels: np.ndarray) -> np.ndarray:
        assert len(mels.shape) == 2
        inputs = {self.onnx_session.get_inputs()[0].name: mels}
        return self.onnx_session.run(None, inputs)[0]

from pathlib import Path
from pprint import pprint
from typing import Literal, Optional

import onnx
import torch
import yaml
from tap import Tap

from ogmios.dataset.commons import DatasetFolder, PreprocessingConfig
from ogmios.layers import Phoneme2Mel
from ogmios.lightning import EfficientSpeech


class OgmiosOnnx(torch.nn.Module):
    def __init__(self,
                 phon2mel: Phoneme2Mel):
        super().__init__()
        self.phon2mel = phon2mel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0)
        mel, lengths = self.phon2mel.synthesize_one(x)
        mel = mel.transpose(1, 2)
        return mel


class ConvertOnnxCommandParser(Tap):
    config: Path  # Path to processing config file (yaml)
    checkpoint: Path  # Model checkpoint that is to be converted to onnx
    verbose: bool = False

    infer_device: Literal['gpu', 'cpu'] = 'cpu'  # Device for which to convert the model
    output_path: Optional[Path] = None


# TODO: add phonemes from dataset to model metadata
# import onnx
# from onnx import helper
#
# # Supposons que votre modèle soit déjà créé
# model = ... # Votre modèle ONNX
#
# # Pour ajouter une liste de chaînes comme métadonnées
# phones = ["a", "b", "c", "d", "e"] # Votre liste de phonèmes
# string_list = helper.StringStringEntryProto(key="phones", value=",".join(phones))
# metadata_props = [string_list]
#
# # Ajouter les métadonnées au modèle
# model.metadata_props.extend(metadata_props)
#
# # Sauvegarder le modèle
# onnx.save(model, "model_with_metadata.onnx")

# main routine
if __name__ == "__main__":
    args = ConvertOnnxCommandParser().parse_args()
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    preprocessing_config = PreprocessingConfig(**config["preprocessing"])
    dataset_folder = DatasetFolder(root_path=Path(config["dataset"]["root_path"]),
                                   ds_name=config["dataset"].get("name"))

    model = EfficientSpeech.load_from_checkpoint(args.checkpoint, map_location=torch.device(args.infer_device))
    model = model.to(args.infer_device)

    ogmios_model = OgmiosOnnx(phon2mel=model.phoneme2mel)


    phoneme = torch.randint(low=0, high=len(dataset_folder.phonemes), size=(69,)).int().to(args.infer_device)
    print("Input shape: ", phoneme.shape)
    sample_input = [phoneme]
    print(f"Converting to {args.checkpoint} to ONNX ...")

    if args.output_path is None:
        output_path = Path("ogmios_onnx.onnx")

    torch.onnx.export(ogmios_model,
                      f=output_path,
                      args=tuple(sample_input),
                      opset_version=18,
                      input_names=["x"],
                      output_names=["mel"],
                      dynamic_axes={
                          "x": {0: "phoneme"},
                          "mel": {0: "frames"}
                      })

    # reload model to add metadata
    print(f"Saving metadata for ONNX model {output_path} ...")
    ogmios_model = onnx.load(output_path)
    phonemes = onnx.StringStringEntryProto(key="phones", value=",".join(dataset_folder.phonemes))
    language = onnx.StringStringEntryProto(key="lang", value=preprocessing_config.text.language)
    metadata_props = [phonemes, language]
    ogmios_model.metadata_props.extend(metadata_props)
    # ogmios_model.metadata_props.extend([onnx.StringStringEntryProto(key="dataset", value=dataset_folder.name)])
    onnx.save(ogmios_model, output_path)

    print("Double checking metadata")
    ogmios_model = onnx.load(output_path)
    metadata = {metadata.key: metadata.value for metadata in ogmios_model.metadata_props}

    if "phones" in metadata:
        metadata["phones"] = metadata["phones"].split(",")
    print("Metadata:")
    print(metadata)
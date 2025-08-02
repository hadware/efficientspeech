- Add support for finetuning
- Notes on finetuning
  - Need for checkpoint retraining
  - Pretrain dataset and finetuning dataset will have different mean/std for energy/pitch
  - This needs to be updated in the Acoustic Decoders

- For validation
  - Have a 2-stage validation
  - One stage, run every x epoch, is to eval just 1 sample from val
  - The other stage, run every 10*x epoch, is to run the full eval on all eval samples

- For mbrola compat:
  - Try out the usage of feature_level: "frame" for pitch
  - this would allow the drop-in usage of .pho files
  - pitch shouldn't be predicted at runtime
  - investigate if speed/pitch height have an effect on mbrola

Notes HIFIGAN:
- hifigan semble vouloir du 128 mel en entrée.
  - Voir à quel point cela alourdi ogmios
  - Voir à quel point le passage à 80 mel dégrade les perfs de vocoding
  - 
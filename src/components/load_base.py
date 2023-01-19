from kfp.v2.dsl import Artifact, Dataset, Input, Model, Output, component

from utils.dependencies import DATASETS, TF_CPU_IMAGE, TORCH, TRANSFORMERS


@component(base_image=TF_CPU_IMAGE,
           packages_to_install=[DATASETS, TRANSFORMERS, TORCH])
def load_base(checkpoint: str, loaded_tokenizer: Output[Artifact],
              loaded_model: Output[Model]) -> None:
    from transformers import (AutoTokenizer,
                              TFAutoModelForSequenceClassification)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = TFAutoModelForSequenceClassification.from_pretrained(
        checkpoint, from_pt=True, force_download=True, resume_download=True)
    tokenizer.save_pretrained(loaded_tokenizer.path)
    model.save_pretrained(loaded_model.path)

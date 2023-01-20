from kfp.v2.dsl import Artifact, Model, Output, component

from utils.dependencies import DATASETS, TF_CPU_IMAGE, TORCH, TRANSFORMERS


@component(base_image=TF_CPU_IMAGE,
           packages_to_install=[DATASETS, TRANSFORMERS, TORCH])
def load_base(checkpoint: str, loaded_tokenizer: Output[Artifact],
              loaded_model: Output[Model]) -> None:
    """
    The load_base function loads a pretrained model and tokenizer from the specified checkpoint.
    The loaded tokenizer is saved to the specified directory, and the loaded model is saved to a
    subdirectory of that directory. The function returns nothing.

    Args:
        checkpoint: str: Specify the path to the model checkpoint
        loaded_tokenizer: Output[Artifact]: Save the tokenizer to a specific location
        loaded_model: Output[Model]: Save the model to a directory

    Returns:
        None
    """
    from transformers import (AutoTokenizer,
                              TFAutoModelForSequenceClassification)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = TFAutoModelForSequenceClassification.from_pretrained(
        checkpoint, from_pt=True, resume_download=True)
    tokenizer.save_pretrained(loaded_tokenizer.path)
    model.save_pretrained(loaded_model.path)

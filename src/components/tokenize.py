from kfp.v2.dsl import Artifact, Dataset, Input, Output, component

from utils.dependencies import DATASETS, TF_CPU_IMAGE, TRANSFORMERS


@component(base_image=TF_CPU_IMAGE,
           packages_to_install=[DATASETS, TRANSFORMERS])
def tokenize(checkpoint: str, interim_dataset: Input[Dataset],
             tokenized_dataset: Output[Dataset],
             loaded_tokenizer: Output[Artifact]) -> None:
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    interim_data = load_from_disk(interim_dataset.path)
    tokenized_data = interim_data.map(
        lambda example: tokenizer(example['main_text'],
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='tf'),
        batched=True)
    tokenized_data.save_to_disk(tokenized_dataset.path)
    tokenizer.save_pretrained(loaded_tokenizer.path)

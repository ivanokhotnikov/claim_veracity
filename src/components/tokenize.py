from kfp.v2.dsl import Artifact, Dataset, Input, Output, component

from utils.dependencies import DATASETS, TF_CPU_IMAGE, TRANSFORMERS


@component(base_image=TF_CPU_IMAGE,
           packages_to_install=[DATASETS, TRANSFORMERS])
def tokenize(interim_dataset: Input[Dataset],
             loaded_tokenizer: Input[Artifact],
             tokenized_dataset: Output[Dataset]) -> None:
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(loaded_tokenizer.path)
    interim_data = load_from_disk(interim_dataset.path)
    tokenized_data = interim_data.map(
        lambda example: tokenizer(example['main_text'],
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='tf'),
        batched=True)
    tokenized_data.save_to_disk(tokenized_dataset.path)

from kfp.v2.dsl import Artifact, Dataset, Input, Output, component

from utils.dependencies import DATASETS, PYTHON310_IMAGE, TRANSFORMERS


@component(base_image=PYTHON310_IMAGE,
           packages_to_install=[DATASETS, TRANSFORMERS])
def tokenize(checkpoint: str, interim_dataset: Input[Dataset],
             tokenized_dataset: Output[Dataset],
             loaded_tokenizer: Output[Artifact]) -> None:
    from datasets import load_from_disk
    from transformers import AutoTokenizer, DataCollatorWithPadding
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    interim_data = load_from_disk(interim_dataset.path)

    def tokenize_fn(example):
        return tokenizer(example['main_text'], padding=False)

    tokenized_data = interim_data.map(tokenize_fn, batched=True)
    tokenized_data.save_to_disk(tokenized_dataset.path)
    tokenizer.save_pretrained(loaded_tokenizer.path)

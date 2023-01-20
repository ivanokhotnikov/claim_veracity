from kfp.v2.dsl import Artifact, Dataset, Input, Output, component

from utils.dependencies import DATASETS, TF_CPU_IMAGE, TRANSFORMERS


@component(base_image=TF_CPU_IMAGE,
           packages_to_install=[DATASETS, TRANSFORMERS])
def tokenize(interim_dataset: Input[Dataset],
             loaded_tokenizer: Input[Artifact],
             tokenized_dataset: Output[Dataset]) -> None:
    """
    The tokenize function takes a dataset of claims and supporting documents,
    and tokenizes them using the BERT tokenizer. The function returns a new dataset
    with the same structure as the original but with tokens instead of text.

    Args:
        interim_dataset: Input[Dataset]: Load the data from disk
        loaded_tokenizer: Input[Artifact]: Load the tokenizer from a previously saved artifact
        tokenized_dataset: Output[Dataset]: Specify the output of this function

    Returns:
        None
    """
    from datasets import load_from_disk
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(loaded_tokenizer.path)
    interim_data = load_from_disk(interim_dataset.path)
    tokenized_data = interim_data.map(
        lambda example: tokenizer(example['claim'],
                                  example['main_text'],
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='tf'),
        batched=True)
    tokenized_data.save_to_disk(tokenized_dataset.path)

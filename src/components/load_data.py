from kfp.v2.dsl import Dataset, Output, component

from utils.dependencies import DATASETS, PYTHON310_IMAGE


@component(base_image=PYTHON310_IMAGE, packages_to_install=[DATASETS])
def load_data(dataset_name: str, interim_data: Output[Dataset]) -> None:
    from datasets import load_dataset
    raw_data = load_dataset(dataset_name)
    interim_dataset = raw_data.filter(lambda x: x['label'] != -1)
    interim_dataset.save_to_disk(interim_data.path)

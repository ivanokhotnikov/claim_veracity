from kfp.v2.dsl import component, Dataset, Output
from utils.dependencies import PYTHON310, DATASETS


@component(base_image=PYTHON310, packages_to_install=DATASETS)
def load_data(dataset_name: str) -> None:
    from datasets import load_dataset
    raw_data = load_dataset(dataset_name)
    interim_data = raw_data.filter(lambda x: x['label'] != -1)
    
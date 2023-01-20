from kfp.v2.dsl import (Artifact, Dataset, Input, Metrics, Model, Output,
                        component)

from utils.dependencies import (DATASETS, GOOGLE_CLOUD_AIPLATFORM, PROTOBUF,
                                TF_CPU_IMAGE, TRANSFORMERS)


@component(base_image=TF_CPU_IMAGE,
           packages_to_install=[
               DATASETS, TRANSFORMERS, GOOGLE_CLOUD_AIPLATFORM, PROTOBUF
           ])
def evaluate(tokenized_dataset: Input[Dataset],
             loaded_tokenizer: Input[Artifact], trained_model: Input[Model],
             test_metrics: Output[Metrics], batch_size: int, timestamp: str,
             project_id: str, location: str, exp_name: str) -> None:
    """
    The evaluate function loads the tokenized dataset,
    loads the trained model, and logs evaluation metrics to AI Platform.

    Args:
        tokenized_dataset: Input[Dataset]: Specify the path to the dataset that has been tokenized by a previous step
        loaded_tokenizer: Input[Artifact]: Load the tokenizer that was used to train the model
        trained_model: Input[Model]: Load the model from disk
        test_metrics: Output[Metrics]: Log the metrics from the evaluation of the model
        batch_size: int: Specify the batch size for evaluation
        timestamp: str: Create a unique run name
        project_id: str: Specify the project_id where your ai platform training and prediction resources are located
        location: str: Specify the region where you want to run your ai platform job
        exp_name: str: Identify the run in ai platform

    Returns:
        None
    """
    import json

    import google.cloud.aiplatform as aip
    from datasets import load_from_disk
    from transformers import (AutoTokenizer, DataCollatorWithPadding,
                              TFAutoModelForSequenceClassification)

    tokenized_data = load_from_disk(tokenized_dataset.path)
    tokenizer = AutoTokenizer.from_pretrained(loaded_tokenizer.path)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,
                                            return_tensors="tf")
    tf_test_dataset = tokenized_data['test'].to_tf_dataset(
        columns=['input_ids', 'attention_mask'],
        label_cols=['label'],
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False)
    model = TFAutoModelForSequenceClassification.from_pretrained(
        trained_model.path)
    results = model.evaluate(tf_test_dataset,
                             batch_size=batch_size,
                             return_dict=True)
    aip.init(project=project_id, location=location, experiment=exp_name)
    aip.start_run(run=timestamp, resume=True)
    aip.log_metrics(results)
    with open(test_metrics.path + '.json', 'w') as metrics_file:
        metrics_file.write(json.dumps(results))
    for k, v in results.items():
        test_metrics.log_metric(k, v)
    aip.end_run()

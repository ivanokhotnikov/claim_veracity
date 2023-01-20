from kfp.v2.dsl import (Artifact, Dataset, Input, Metrics, Model, Output,
                        component)

from utils.dependencies import (DATASETS, GOOGLE_CLOUD_AIPLATFORM, PROTOBUF,
                                TF_CPU_IMAGE, TORCH, TRANSFORMERS)


@component(base_image=TF_CPU_IMAGE,
           packages_to_install=[
               DATASETS, TRANSFORMERS, TORCH, GOOGLE_CLOUD_AIPLATFORM, PROTOBUF
           ])
def train(tokenized_dataset: Input[Dataset], loaded_tokenizer: Input[Artifact],
          loaded_model: Input[Model], trained_model: Output[Model],
          train_metrics: Output[Metrics], batch_size: int, epochs: int,
          learning_rate: float, timestamp: str, project_id: str, location: str,
          exp_name: str, decay_rate: float) -> None:
    """
    The train function trains a model on the provided dataset.

    Args:
        tokenized_dataset: Input[Dataset]: Pass the tokenized dataset
        loaded_tokenizer: Input[Artifact]: Load the tokenizer from a previously trained model
        loaded_model: Input[Model]: Load a pretrained model from the gcs bucket
        trained_model: Output[Model]: Store the trained model
        train_metrics: Output[Metrics]: Log the training metrics to ai platform
        batch_size: int: Specify the number of samples in each batch
        epochs: int: Specify the number of epochs to train for
        learning_rate: float: Set the initial learning rate
        timestamp: str: Create a unique run name
        project_id: str: Specify the project id where the ai platform training and prediction resources will be created
        location: str: Specify the region where your ai platform training runs are executed
        exp_name: str: Name the ai platform experiment
        decay_rate: float: Control the exponential decay rate of the learning rate

    Returns:
        None
    """
    import json

    import google.cloud.aiplatform as aip
    from datasets import load_from_disk
    from keras.losses import SparseCategoricalCrossentropy
    from keras.optimizers import Adam
    from keras.optimizers.schedules.learning_rate_schedule import \
        ExponentialDecay
    from transformers import (AutoTokenizer, DataCollatorWithPadding,
                              TFAutoModelForSequenceClassification)
    tokenized_data = load_from_disk(tokenized_dataset.path)
    tokenizer = AutoTokenizer.from_pretrained(loaded_tokenizer.path)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer,
                                            return_tensors="tf")
    tf_train_dataset = tokenized_data['train'].to_tf_dataset(
        columns=['input_ids', 'attention_mask'],
        label_cols=['label'],
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=True)
    tf_validation_dataset = tokenized_data['validation'].to_tf_dataset(
        columns=['input_ids', 'attention_mask'],
        label_cols=['label'],
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False)
    num_train_steps = len(tf_train_dataset) * epochs
    lr_scheduler = ExponentialDecay(initial_learning_rate=learning_rate,
                                    decay_steps=num_train_steps,
                                    decay_rate=decay_rate,
                                    staircase=True)
    model = TFAutoModelForSequenceClassification.from_pretrained(
        loaded_model.path)
    model.compile(optimizer=Adam(learning_rate=lr_scheduler),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(tf_train_dataset,
                        validation_data=tf_validation_dataset,
                        epochs=epochs)
    aip.init(experiment=exp_name, project=project_id, location=location)
    aip.start_run(run=timestamp)
    model.save(trained_model.path + '.h5')
    with open(train_metrics.path + '.json', 'w') as metrics_file:
        metrics_file.write(json.dumps(history.history))
    for k, v in history.history.items():
        history.history[k] = [float(vi) for vi in v]
        train_metrics.log_metric(k, history.history[k])
        aip.log_metrics(k, v)
    aip.log_params({
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'decay_rate': decay_rate
    })

from kfp.v2.dsl import Artifact, Dataset, Input, Output, component

from utils.dependencies import (DATASETS, GOOGLE_CLOUD_AIPLATFORM, PROTOBUF,
                                TF_CPU_IMAGE, TORCH, TRANSFORMERS)


@component(base_image=TF_CPU_IMAGE,
           packages_to_install=[
               DATASETS, TRANSFORMERS, TORCH, GOOGLE_CLOUD_AIPLATFORM, PROTOBUF
           ])
def train(tokenized_dataset: Input[Dataset], loaded_tokenizer: Input[Artifact],
          batch_size: int, epochs: int, learning_rate: float, checkpoint: str,
          timestamp: str, project_id: str, location: str, exp_name: str,
          decay_rate: float) -> None:
    import google.cloud.aiplatform as aip
    from datasets import load_from_disk
    from keras.losses import SparseCategoricalCrossentropy
    from keras.optimizers import Adam
    from keras.optimizers.schedules.learning_rate_schedule import \
        ExponentialDecay
    from transformers import (AutoTokenizer, DataCollatorWithPadding,
                              TFAutoModelForSequenceClassification)
    tokenizer = AutoTokenizer.from_pretrained(loaded_tokenizer.path)
    tokenized_data = load_from_disk(tokenized_dataset.path)
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer,
    #                                         return_tensors='tf')
    tf_train_dataset = tokenized_data['train'].to_tf_dataset(
        columns=['input_ids', 'attention_mask'],
        label_cols=['label'],
        batch_size=batch_size,
        shuffle=True)
    tf_validation_dataset = tokenized_data['validation'].to_tf_dataset(
        columns=['input_ids', 'attention_mask'],
        label_cols=['label'],
        batch_size=batch_size,
        shuffle=False)
    num_train_steps = len(tf_train_dataset) * epochs
    lr_scheduler = ExponentialDecay(initial_learning_rate=learning_rate,
                                    decay_steps=num_train_steps,
                                    decay_rate=decay_rate,
                                    staircase=True)
    model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint,
                                                                 from_pt=True)
    model.compile(optimizer=Adam(learning_rate=lr_scheduler),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(tf_train_dataset,
                        validation_data=tf_validation_dataset,
                        epochs=epochs)
    aip.init(experiment=exp_name, project=project_id, location=location)
    aip.start_run(run=timestamp)
    aip.log_params({
        'batch_size': batch_size,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'decay_rate': decay_rate
    })
    aip.log_metrics({
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    })

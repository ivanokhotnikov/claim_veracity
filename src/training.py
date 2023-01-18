import argparse

from datasets import load_dataset
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay
from transformers import (AutoTokenizer, DataCollatorWithPadding,
                          TFAutoModelForSequenceClassification, Trainer)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--train_batch_size', type=int)
    parser.add_argument('--val_batch_size', type=int)
    parser.add_argument('--epochs', type=int)
    return parser.parse_args()


train_batch_size = 8
val_batch_size = 16
epochs = 4

dataset = 'health_fact'
raw_data = load_dataset(dataset)
interim_data = raw_data.filter(lambda x: x['label'] != -1)

distilbert_checkpoint = 'distilbert-base-uncased-finetuned-sst-2-english'
longformer_checkpoint = 'nbroad/longformer-base-health-fact'

tokenizer = AutoTokenizer.from_pretrained(longformer_checkpoint)
model = TFAutoModelForSequenceClassification.from_pretrained(
    longformer_checkpoint, from_pt=True)


def tokenize_fn(example):
    return tokenizer(example['main_text'], padding=False)


tokenized_data = interim_data.map(tokenize_fn, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer,
                                        return_tensors='tf')
tf_train_dataset = tokenized_data['train'].to_tf_dataset(
    columns=['input_ids', 'attention_mask'],
    label_cols=['labels'],
    batch_size=train_batch_size,
    collate_fn=data_collator,
    shuffle=True)
tf_validation_dataset = tokenized_data['validation'].to_tf_dataset(
    columns=['input_ids', 'attention_mask'],
    label_cols=['labels'],
    batch_size=val_batch_size,
    collate_fn=data_collator,
    shuffle=False)

num_train_steps = len(tf_train_dataset) * epochs
lr_scheduler = ExponentialDecay(initial_learning_rate=5e-5,
                                decay_steps=num_train_steps,
                                decay_rate=0.96,
                                staircase=True)
model.compile(optimizer=Adam(learning_rate=lr_scheduler),
              loss=SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(tf_train_dataset,
          validation_data=tf_validation_dataset,
          epochs=epochs)

if __name__ == '__main__':
    args = vars(get_arguments)

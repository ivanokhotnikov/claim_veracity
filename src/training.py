import argparse
import os
from datetime import datetime

import google.cloud.aiplatform as aip
from dotenv import load_dotenv
from kfp.v2 import compiler
from kfp.v2.dsl import pipeline

from components import evaluate, load_base, load_data, tokenize, train


@pipeline(name='training')
def training_pipeline(dataset_name: str, checkpoint: str, batch_size: int,
                      epochs: int, learning_rate: float, exp_name: str,
                      timestamp: str, project_id: str, location: str,
                      decay_rate: float):
    load_data_task = load_data(dataset_name=dataset_name)
    load_base_task = load_base(checkpoint=checkpoint)
    tokenize_task = tokenize(
        loaded_tokenizer=load_base_task.outputs['loaded_tokenizer'],
        interim_dataset=load_data_task.outputs['interim_data'])
    train_task = (train(
        timestamp=timestamp,
        project_id=project_id,
        location=location,
        exp_name=exp_name,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        decay_rate=decay_rate,
        loaded_tokenizer=load_base_task.outputs['loaded_tokenizer'],
        tokenized_dataset=tokenize_task.outputs['tokenized_dataset'],
        loaded_model=load_base_task.outputs['loaded_model'])
                  ).set_cpu_limit('8').set_memory_limit('64G')
    evaluate_task = (evaluate(
        timestamp=timestamp,
        project_id=project_id,
        location=location,
        exp_name=exp_name,
        batch_size=batch_size,
        loaded_tokenizer=load_base_task.outputs['loaded_tokenizer'],
        tokenized_dataset=tokenize_task.outputs['tokenized_dataset'],
        trained_model=train_task.outputs['trained_model'])
                     ).set_cpu_limit('8').set_memory_limit('64G')


def get_arguments():
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument('--exp_name', type=str, default='default')
    parser.add_argument('--checkpoint',
                        type=str,
                        default='nbroad/longformer-base-health-fact')
    parser.add_argument('--dataset_name', type=str, default='health_fact')
    # Training
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1)
    parser.add_argument('--decay_rate', type=float, default=0.96)
    return parser.parse_args()


if __name__ == '__main__':
    args = vars(get_arguments())
    load_dotenv()
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    os.makedirs('./compiled', exist_ok=True)
    compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path=f'./compiled/training_{timestamp}.json')
    aip.init(project=os.environ['PROJECT_ID'],
             location=os.environ['LOCATION'],
             staging_bucket=os.environ['PIPELINES_URI'])
    job = aip.PipelineJob(
        enable_caching=True,
        display_name=timestamp,
        pipeline_root=os.environ['PIPELINES_URI'],
        template_path=f'./compiled/training_{timestamp}.json',
        parameter_values={
            'timestamp': timestamp,
            'project_id': os.environ['PROJECT_ID'],
            'location': os.environ['LOCATION'],
            **args
        })
    job.submit()

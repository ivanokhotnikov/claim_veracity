# Claim veracity checker

The repository contains the source code for the training and inference of the claim veracity machine learning system. The system is built upon the longformer tuned to PUBHEALTH dataset. The training pipeline allows to further fine tune the longformer (or any other model defined by the `checkpoint` training parameter) to any dataset defined by the `dataset_name` training parameter.

## Base model

[The base model](nbroad/longformer-base-health-fact) used in training is derived from the [longormer architecture](https://arxiv.org/abs/2004.05150).

## Dataset

[Dataset](https://huggingface.co/datasets/health_fact)

## Hugging face demo

[Demo](https://huggingface.co/spaces/ivanokhotnikov/longformer-base-health-fact)

## Prerequisites

 - Python 3.10
 - [Google Cloud CLI](https://cloud.google.com/sdk/docs/install-sdk)

## Setup

In the repository, execute:

1. `sh setup.sh`

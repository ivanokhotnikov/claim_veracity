# Claim veracity checker

The repository contains the source code for the training and inference of the claim veracity machine learning system. The system is built upon the longformer tuned to PUBHEALTH dataset. The training pipeline allows to further fine tune the longformer (or any other model defined by the `checkpoint` training parameter) to any dataset defined by the `dataset_name` training parameter.

## Base model

[The base model](nbroad/longformer-base-health-fact) used in training is derived from the [longormer architecture](https://arxiv.org/abs/2004.05150).

## Dataset

The base longformer as well development version were tuned on [PUBHEALTH dataset](https://huggingface.co/datasets/health_fact)

## Hugging face demo

[Demo](https://huggingface.co/spaces/ivanokhotnikov/longformer-base-health-fact)

## Scalable GKE demo

[Demo](http://130.211.14.19)

## Prerequisites

 - Python 3.10

## Setup

To set the environment `sh setup.sh`

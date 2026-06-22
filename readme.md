# Embedding alignment model
Model used as a part of my master's thesis project

## Description
The goal of the pipeline is to yield a speech-to-retrieval type model by combining text and speech modalities. Due to original work operating in a very restricted domain (Polish medical speech), it uses Data2Vec style pretraining in order to facilitate the usage of unsupervised datasets, both text and speech. It then combines 2 separate pretrained encoders into one final embedding model, allowing for intermodal search between text and speech

## Running the unsupervised training
Unsupervised training is ran via configuration files with `train.py` script. You can see example configs in `configs/example` directory. All the run info, including tensorboard logs, config and checkpoints are saved into its own experiment directory in `runs`.

`train.py` allows for splitting the training into 2 threads operating on separate gpus. First thread runs the training feature encoder with optional augementations, streaming into a queue consumed by the training process.

## TODO
- add augmentation to the training encoder pipeline, ran via configs
- support model loading from the previous experiments, for checkpointing and evaluation
- add finetuning process for final embedding alignment

# Semi-supervised learning research - Fall 2020

## Install dependecies

### For running the models

Install [pytorch](https://pytorch.org/) and [allenlp](https://github.com/allenai/allennlp).

For hyperparameter tunning install [optuna](https://github.com/optuna/optuna) and [optuna dashboard](https://github.com/optuna/optuna-dashboard)

### For data augmentation

Install [NLP AUG](https://github.com/makcedward/nlpaug).

## Allen NLP

### How to run experiments

`allennlp train [config_file_path] -s [results-dir]`

### How to use the models

`allennlp predict [model_dir]/model.tar.gz [data_file_path].json --predictor [predictor_registered_name]`

## TensorBoard

### How to show results in tensorboard

`tensorboard --logdir=[log_dir]`

### How to upload results

` tensorboard dev upload --logdir=[log_dir]`

# Semi-supervised learning research - Fall 2020

## Install dependecies

### For running the models

Install [pytorch](https://pytorch.org/) and [allenlp](https://github.com/allenai/allennlp).

For hyperparameter tunning install [optuna](https://github.com/optuna/optuna) and [optuna dashboard](https://github.com/optuna/optuna-dashboard).

### For data augmentation

Install [NLP AUG](https://github.com/makcedward/nlpaug), and related libraries (They can be found at the same github repo).

## Train models with Allen NLP

### How to run experiments

`allennlp train [config_file_path] -s [results-dir]`

### How to use the models

`allennlp predict [model_dir]/model.tar.gz [data_file_path].json --predictor [predictor_registered_name]`

## Visualize the results with TensorBoard

### How to show results locally

`tensorboard --logdir=[log_dir]`

### How to upload results

`tensorboard dev upload --logdir=[log_dir]`

## Perform hyperparameter optimization with Optuna

### How to run the optimization 

Run file containing optuna code: `python hyperparamet_optim.py`

### How to vizualize the optimization

`optuna-dashboard sqlite:[path-to-.db-file]`

## Repo Structure

### [experiments](experiments)

All the data, models and results are within this directory. For more information read the particular [README](experiments/README.md) for this directory.


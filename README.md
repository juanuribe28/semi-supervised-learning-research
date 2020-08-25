# Semi-supervised learning research - Fall 2020

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

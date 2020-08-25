# Semi-supervised learning research - Fall 2020

## How to run experiments

`allennlp train [config_file_path] -s [results-dir]`

## How to use the models

`allennlp predict [model_dir]/model.tar.gz [data_file_path].json --predictor [predictor_registered_name]

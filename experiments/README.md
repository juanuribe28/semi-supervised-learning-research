# experiments directory structure

## Hyperparameter optimizer

-**hyperparam-optim.py:** Python code used to run Optuna optimizer on a specific model.
-**hyperparam-optim.db:** Contains all the relevant information from the different hyperparameter optimization tests with optuna (Recommended to open with optuna dashboard for optimal visualization).

### [data](data)

Contains all the data used for the experiments, including the python files used for data augmentation and analyzing results.For more information check the [README](data/README.md).

### [template-exp](template-exp)

Template repo for an experiment with AllenNLP (Note that most files in this directory are empty, and are just meant to show the general structure. For working examples refer to actual experiment directories). It contains:

- **.allennlp_plugins:** necessary file to run allennlp models. Contains the path to the model. No need to modify.
- **architecture:** Contains the allennlp modeland related classes.
    - **dataset_reader.py:** Reads and loads the data do be used by the model. A DatasetReader class.
    - **model.py:** The actual AllenNLP model. A Model class.
    - **predictor.py:** Predicts results using a trained model. A Predictor class.
- **training_config:** Contains the jsonnet file with the specific configuration for the model.
- **results:** Output directory for all the test results.

## Old experiment directories

There's a specific directory for each type of model. Recommended only to review previous results, not to update or to work with. Instead use new universal model experiment.

### [sentence-exp](sentence-exp)

Contains model and results using the sentence as input.

### [verb-sent-exp](verb-sent-exp)

Contains model and results using only the tagged excercise as input.

### [verb-sentence-exp](verb-sentence-exp)

Contains model and results using both the full sentence and the tagged exercise as input.

## New unversal model

### [universal-exp](universal-exp)

Contains model for using both the full sentece and/or the tagged exercise as input. Instead of having to use a specific model for different types of inputs as in previous experiments, this model allows to set up the weight of each input as a hyperparameter.

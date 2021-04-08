## experiments directory structure

**hyperparam-optim.db** contains all the relevant information from the different hyperparameter optimization tests with optuna (Recommended to open with optuna dashboard for optimal visualization).

### [data](data)

### [template-exp](template-exp)

Template repo for an experiment with AllenNLP (Note that most files in this directory are empty, and are just meant to show the general structure. For working examples refer to actual experiment directories). It contains:

- **.allennlp_plugins:** necessary file to run allennlp models. Contains the path to the model. No need to modify.
- **data:** 
- **architecture:** Contains the allennlp modeland related classes.
    - **dataset_reader.py:** Reads and loads the data do be used by the model. A DatasetReader class.
    - **model.py:** The actual AllenNLP model. A Model class.
    - **predictor.py:** Predicts results using a trained model. A Predictor class.
- **training_config:** Contains the jsonnet file with the specific configuration for the model.
- **results:** Output directory for all the test results.


### [sentence-exp](sentence-exp)

### [verb-sent-exp](verb-sent-exp)

### [verb-sentence-exp](verb-sentence-exp)

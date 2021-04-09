# data directory structure

## Original data

Original excercise data in json format collected in previous research. For more information check [this repo](https://github.com/mayaepps/exercise-logs).

-**trainDataValues.json:** Original training set containing 2400 examples.
-**testDataValues.json:** Original testing set containing 600 examples.

## Sorted data

Using [extract_json_to_tsv.py](extract_json_to_tsv.py) and [order.py](order.py) we extracted all the values from the original data and divided it into new train and test groups that would be balanced for eache label. 

- **all_data.tsv:** Contains all 3000 examples.
- **train_data.tsv:** Training set with 2393 examples.
- **test_data.tsv:** Testing set with 604 examples.

Note: train and test set have removed duplicates and labels without at least 2 examples have been changed to other.

## Data augmentation

Using [increase_data.py](increase_data.py) we preformed data augmentation on the sorted data.

-**aug_train_data#.tsv:** Results from data augmentation. Each file went through a different augmentation process. For information regarding the specific configuration of each file, refer to [this document](https://docs.google.com/document/d/15X_Z53kOcll3FDYFXh5T6eYpFUOO9DynMWc-Onk0nJw/edit?usp=sharing).

## Data analysis

Python files that help analyzing the results from the models.

- **analyze_general_results.py:** Analyzes the result of a trained model with respect to the training and testing set. It compares the number of training examples for each label with the respective training and testing accuracy, and the number of testing examples. 
- **analyze_particular_results.py:** Analyzes the results of a trained model for each labeled example in the testing and training set. It checks the specific accuracy for each labeled and it generates a table with the of correct and incorrect examples for each label. It generetas two excel files. One with the complete table and one with the summary.

### [analysis](analysis)

Output directory for the anlysis files. 

## Data prunning

After analyzing the results from the best data augmentation runs, the follwing changes were made to the ordered datasets:
    - Weight flies -> Weights, General
    - Water Aerobics -> Aerobics
    - Power walking -> Walking
    - Leg Curls -> Leg Raises
    - Manually fixed Squash and Sports, general

-**train_data_reduced.tsv:** Training set with 2393 examples.
-**test_data_reduced.tsv:** Testing set with 604 examples.

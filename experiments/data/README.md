# data directory structure

## Original data

Original excercise data in json format collected in previous research. Formore information check this [repo](https://github.com/mayaepps/exercise-logs).

-**trainDataValues.json:** Original training set containing 2400 examples.
-**testDataValues.json:** Original testing set containing 600 examples.

## Sorted data

Using [extract_json_to_tsv.py](extract_json_to_tsv.py) and [order.py](order.py) we extracted all the values from the original data and divided it into new train and test groups that would be balanced for eache label. 

-**all_data.tsv:** Contains all 3000 examples.
-**train_data.tsv:** Training set with 2393 examples.
-**test_data.tsv:** Testing set with 604 examples.

Note: train and test set have removed duplicates and labels without at least 2 examples have been changed to other.

## Data augmentation

Using [increase_data.py](increase_data.py)

-**aug_train_data#.tsv:**

## Data prunning

-**train_data_reduced.tsv:**
-**test_data_reduced.tsv:**

## Data analysis

-**analyze_model_results.py:**
-**analyze_aug_results.py:**

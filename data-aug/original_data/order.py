from typing import Dict, Set, Tuple, List
from copy import deepcopy

import matplotlib.pyplot as plt
import csv
import sys

DataDict = Dict[str, Set[str]]

def update_examples_dict(data_path: str, data_dict: DataDict = dict()) -> DataDict:
    with open(data_path) as lines:
        for line in lines:
            sent, label = line.replace('\n', '').split('\t')
            if label in data_dict:
                data_dict[label].add(sent)
            else:
                data_dict[label] = set([sent])
    return data_dict

def remove_single_examples(data_dict: DataDict) -> Tuple[DataDict, DataDict]:
    removed_examples = dict()
    for label, sent_set in data_dict.copy().items():
        if len(sent_set) == 1:
            removed_examples[label] = data_dict.pop(label)
    return data_dict, removed_examples

def divide_data(data_dict: DataDict, test_percent: int = 0.2) -> Tuple[DataDict, DataDict]:
    training_set, testing_set = dict(), dict()
    for label, sent_set in data_dict.items():
        n_test_sents = max(round(len(sent_set) * 0.2), 1)
        testing_set[label] = {sent_set.pop() for x in range(n_test_sents)}
        training_set[label] = sent_set
    return training_set, testing_set

def add_examples_to_other(data_dict: DataDict, other_examples: List) -> DataDict:
    for other_example in other_examples:
        data_dict['Other'].add(other_example)
    return data_dict

def count_items(data_dict: DataDict) -> int:
    total_data = 0
    for label, sents_set in data_dict.items():
        total_data += len(sents_set)
    return total_data

def save_tsv_data(datasets: List[DataDict], file_names: List[str], exp_dir: str = '') -> None:
    for dataset, file_name in zip(datasets, file_names):
        with open('./{}new_{}_data.tsv'.format(exp_dir, file_name), 'w', newline='', encoding='utf8') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            for label, sent_set in dataset.items():
                for sent in sent_set:
                    tsv_writer.writerow([sent, label])
    return

def get_exp_dir() -> str:
    if len(sys.argv) != 2:
        raise Exception('Missing experiment directory name')
    exp_dir = sys.argv[1]

    if exp_dir[-1] != '/':
        exp_dir = '{}/'.format(exp_dir)
    return exp_dir

if __name__ == "__main__":

    exp_dir = get_exp_dir()

    test_data_path = './{}test_data.tsv'.format(exp_dir)
    train_data_path = './{}train_data.tsv'.format(exp_dir)

    all_data = update_examples_dict(test_data_path)
    all_data = update_examples_dict(train_data_path, all_data)
    
    repeated_data, removed_data = remove_single_examples(deepcopy(all_data))

    removed_examples = [example for example_set in removed_data.values() for example in example_set]

    full_data = add_examples_to_other(deepcopy(repeated_data), removed_examples)

    train_data, test_data = divide_data(deepcopy(full_data), 600 / 3000)
    
    save_tsv_data([all_data, train_data, test_data], ['all', 'train', 'test'], exp_dir=exp_dir)

    fig1, ax1 = plt.subplots()

    n_train_examples = [len(example_set) for example_set in train_data.values()]
    n_test_examples = [len(example_set) for example_set in test_data.values()]
    ax1.scatter(n_train_examples, n_test_examples)
    ax1.set_title('Training examples vs Testing examples')
    fig1.show()
    plt.show()

import nlpaug.augmenter.char as char_aug
import nlpaug.augmenter.word as word_aug
import nlpaug.augmenter.sentence as sentence_aug
import nlpaug.flow as aug_flow
import pandas as pd
import sys
import random
import os
import csv
from tqdm import tqdm
from typing import Dict, Set, Tuple, List

from nlpaug.util import Action
from copy import deepcopy

DataDict = Dict[str, Set[str]]

def update_examples_dict(data_path: str, data_dict: DataDict = dict()) -> DataDict:
    with open(data_path) as lines:
        for line in lines:
            sentence, verb, label = line.replace('\n', '').split('\t')
            if label in data_dict:
                data_dict[label].add((sentence))
            else:
                data_dict[label] = set([sentence])
    return data_dict

def get_max_set_len(data_dict: DataDict) -> int:
    return max(map(len, data_dict.values()))

def augment_data_dict(data_dict: DataDict, max_set_len: int, ratio: int = 1) -> None:
    new_len = round(max_set_len * ratio)
    for label, dataset in tqdm(data_dict.items()):
        original_dataset = list(deepcopy(dataset))
        while len(dataset) < min(new_len, len(original_dataset) * 1.5):
            dataset.add(augment_sentence(random.choice(original_dataset)))

def augment_sentence(sentence: str) -> str:
    aug = aug_flow.Sequential([
       word_aug.ContextualWordEmbsAug(model_path='xlnet-base-cased', action="substitute", device="cuda"),
       word_aug.ContextualWordEmbsAug(model_path='xlnet-base-cased', action="insert", device="cuda"),
       word_aug.RandomWordAug(action="swap"),
       word_aug.SpellingAug(),
       char_aug.KeyboardAug(),
    ])
    return aug.augment(sentence)

def save_as_tsv(dataset: DataDict, path: str) -> None:
    with open(path, 'w', newline='', encoding='utf8') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for label, sent_set in dataset.items():
            for sent in sent_set:
                tsv_writer.writerow([sent, label])
    return

if __name__ == "__main__":
    path = './data/train_data.tsv'

    data = update_examples_dict(path)
    max_len = get_max_set_len(data)
    augment_data_dict(data, max_len, 1.1)
    save_as_tsv(data, './data/aug_train_data7.tsv')

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

from nlpaug.util import Action
from original_data.order import DataDict
from original_data import order

from copy import deepcopy


def get_paths() -> str:
    if len(sys.argv) != 2:
        raise Exception('Missing experiment directory name')
    exp_dir = sys.argv[1]

    if exp_dir[-1] != '/':
        exp_dir = '{}/'.format(exp_dir)
    return exp_dir

def get_max_set_len(data_dict: DataDict) -> int:
    return max(map(len, data_dict.values()))

def augment_data_dict(data_dict: DataDict, max_set_len: int, ratio: int = 1) -> None:
    new_len = round(max_set_len * ratio)
    for label, dataset in tqdm(data_dict.items()):
        original_dataset = list(deepcopy(dataset))
        while len(dataset) < min(new_len, len(original_dataset) * 5):
            dataset.add(augment_sentence(random.choice(original_dataset)))

def augment_sentence(sentence: str) -> str:
    aug = aug_flow.Sequential([
        word_aug.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert"),
        word_aug.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute"),
        word_aug.ContextualWordEmbsAug(model_path='roberta-base', action="substitute"),
        word_aug.SynonymAug(aug_src='wordnet'),
        word_aug.RandomWordAug(action="swap"),
        word_aug.SpellingAug(),
        # word_aug.BackTranslationAug(from_model_name='transformer.wmt19.en-de', 
        # to_model_name='transformer.wmt19.de-en'),
        char_aug.KeyboardAug(),
        char_aug.RandomCharAug(action='swap'),
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
    path = './original_data/new_train_data.tsv'

    data = order.update_examples_dict(path)
    max_len = get_max_set_len(data)
    augment_data_dict(data, max_len, 1.1)
    save_as_tsv(data, './original_data/train_data_aug(seq-5x-big).tsv')

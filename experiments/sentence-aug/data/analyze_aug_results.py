from allennlp.predictors.predictor import Predictor

import sys
import pandas as pd
from pandas import DataFrame

if __name__ == '__main__':
    if len(sys.argv) == 2:
        save_name = sys.argv[1]
        k = 1
    elif len(sys.argv) == 3:
        save_name = sys.argv[1]
        k = int(sys.argv[2])
    else:
        print('analyze_aug_results.py <output file name> [top-k value = 1]')
        sys.exit(2)

    model_path = './results/sequential-5xLimit/model.tar.gz'
    data_paths = { 'train' : './data/train_data_aug(seq-5x).tsv', 
                   'test'  : './data/new_test_data.tsv'}

    predictor = Predictor.from_path(model_path, predictor_name='topk_exercise_classifier')

    data_names = ['train', 'test']
    column_names = ['total', 'correct', 'incorrect', 'correct_exp',  'incorrect_exp', 'incorrect_labeled_exp', 'acc']
    columns = pd.MultiIndex.from_product([data_names, column_names])

    vocab_labels = predictor._model.vocab.get_token_to_index_vocabulary('labels')
    labels_dict = {label: [0, 0, 0, list(), list(), list(), 0, 0, 0, 0, list(), list(), list(), 0] for label in vocab_labels}
    
    results_df = DataFrame.from_dict(labels_dict, 
                                     orient = 'index',
                                     columns = columns)
    totals_df = pd.DataFrame(results_df.sum())

    for data_name in data_names:
        data_path = data_paths[data_name]
        data_df = pd.read_csv(data_path, sep='\t', header=None)
        for _, data_row in data_df.iterrows():
            sent, label = [x for x in data_row]

            results_df.loc[label, (data_name, 'total')] += 1

            predictions = predictor.dump_line(predictor.predict(sent), k=k)
            predicted_labels = list(zip(*predictions))[0]
            predicted_label = predicted_labels[0]

            if label in predicted_labels:
                results_df.loc[label, (data_name, 'correct_exp')].append(sent)
                results_df.loc[label, (data_name, 'correct')] += 1
            else:
                results_df.loc[predicted_label, (data_name, 'incorrect_exp')].append((sent, label))
                results_df.loc[predicted_label, (data_name, 'incorrect')] += 1
                results_df.loc[label, (data_name, 'incorrect_labeled_exp')].append(((sent, predicted_labels)))
            
            results_df.loc[label, (data_name, 'acc')] = (results_df.loc[label, (data_name, 'correct')] 
                                              / results_df.loc[label, (data_name, 'total')]) * 100

        totals_df.loc[(data_name, 'total')] = results_df.sum().loc[(data_name, 'total')]
        totals_df.loc[(data_name, 'correct')] = results_df.sum().loc[(data_name, 'correct')]
        totals_df.loc[(data_name, 'acc')] = (totals_df.loc[(data_name, 'correct')] 
                                              / totals_df.loc[(data_name, 'total')]) * 100

    results_df.to_excel('./data/{}.xlsx'.format(save_name)) 
    totals_df.to_excel('./data/{}_sum.xlsx'.format(save_name))

    print(results_df)
    print(totals_df)

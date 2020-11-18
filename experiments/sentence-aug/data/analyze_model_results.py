from allennlp.predictors.predictor import Predictor

import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame

if __name__ == '__main__':
    
    model_path = './results/sequential-5xLimit-big/model.tar.gz'
    predictor = Predictor.from_path(model_path, predictor_name='exercise_classifier')

    data_names = ['train', 'test']
    column_names = ['total', 'correct', 'incorrect', 'acc']
    columns = pd.MultiIndex.from_product([data_names, column_names])
    n_columns = len(data_names) * len(column_names)
    vocab_labels = predictor._model.vocab.get_token_to_index_vocabulary('labels')
    labels_dict = {label: [0 for x in range(n_columns)] for label in vocab_labels}
    results_df = DataFrame.from_dict(labels_dict, 
                                     orient = 'index',
                                     columns = columns)
    totals_df = pd.DataFrame(results_df.sum())

    data_paths = ['./data/train_data_aug(seq-5x-big).tsv', './data/new_test_data.tsv']
    for data_name, data_path in zip(data_names, data_paths):
        data_df = pd.read_csv(data_path, sep='\t', header=None)
        for _, data_row in data_df.iterrows():
            sent, label = [x for x in data_row]
            results_row = results_df.loc[label]
            results_row[(data_name, 'total')] += 1

            prediction = predictor.dump_line(predictor.predict(sent))
            predicted_label = prediction[0][0]
            
            if label == predicted_label:
                results_row[(data_name, 'correct')] += 1 
            else:
                results_df.loc[predicted_label, (data_name, 'incorrect')] += 1
            
            results_row[(data_name, 'acc')] = (results_row[(data_name, 'correct')] 
                                              / results_row[(data_name, 'total')]) * 100
            
        totals_df.loc[(data_name, 'total')] = results_df.sum().loc[(data_name, 'total')]
        totals_df.loc[(data_name, 'correct')] = results_df.sum().loc[(data_name, 'correct')]
        totals_df.loc[(data_name, 'acc')] = (totals_df.loc[(data_name, 'correct')] 
                                              / totals_df.loc[(data_name, 'total')]) * 100
        
    fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    ax1.scatter(results_df.loc[:, ('train', 'total')], results_df.loc[:, ('train', 'acc')])
    ax1.set_title('Training examples vs Training acc')
    
    ax2.scatter(results_df.loc[:, ('train', 'total')], results_df.loc[:, ('test', 'acc')])
    ax2.set_title('Training examples vs Testing acc')
    fig1.show()

    fig2, ax3 = plt.subplots()	
    ax3.scatter(results_df.loc[:, ('train', 'total')], results_df.loc[:, ('test', 'total')])	
    ax3.set_title('Training examples vs Testing examples')	
    fig2.show()

    plt.show()

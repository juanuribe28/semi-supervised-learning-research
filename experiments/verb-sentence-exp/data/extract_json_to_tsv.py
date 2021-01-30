import json
import csv
import sys

if __name__ == '__main__':

    if len(sys.argv) != 2:
        raise Exception('Missing output directory name')
    exp_dir = sys.argv[1]

    if exp_dir[-1] != '/':
        exp_dir = '{}/'.format(exp_dir)

    curly_apostrophe = '‰Ûª'

    datasets = ['train', 'test']
    with open('./{}all_data.tsv'.format(exp_dir), 'w', newline='', encoding='utf8') as out_file:
        for dataset in datasets:
            data = json.load(open('./data/{}DataValues.json'.format(dataset), encoding='utf8'))
            tsv_writer = csv.writer(out_file, delimiter='\t')
            for sent_id in data:
                sentence = data[sent_id]['sentence'].replace('\n', '')
                verb = data[sent_id]['exercise segment'].replace('\n', '')
                label = data[sent_id]['exercise value'].replace('\n', '')
                if label == 'Chest press':
                    label = 'Bench press'
                elif label == 'Body weight exercises, general':
                    label = 'Bodyweight exercises, general'
                tsv_writer.writerow([sentence.replace(curly_apostrophe, "'"), 
                                    verb.replace(curly_apostrophe, "'"), 
                                    label])
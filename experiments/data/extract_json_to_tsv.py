import json
import csv
import sys

if __name__ == '__main__':

    if len(sys.argv) != 2:
        raise Exception('Missing output directory name')
    exp_dir = sys.argv[1]

    if exp_dir[-1] != '/':
        exp_dir = '{}/'.format(exp_dir)

    sets = ['train', 'test']
    for set in sets:
        data = json.load(open('./data/{}DataValues.json'.format(set)))
        with open('./{}{}_data.tsv'.format(exp_dir, set), 'w', newline='') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            for sent_id in data:
                sent = data[sent_id]['sentence'].replace('\n', '')
                label = data[sent_id]['exercise value'].replace('\n', '')
                tsv_writer.writerow([sent, label])
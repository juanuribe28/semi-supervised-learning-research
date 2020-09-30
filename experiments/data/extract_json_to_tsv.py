import json
import csv

if __name__ == '__main__':

    sets = ['train', 'test']

    for set in sets:
        data = json.load(open('./{}DataValues.json'.format(set)))
        with open('./{}_data.tsv'.format(set), 'w', newline='') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            for sent_id in data:
                sent = data[sent_id]['sentence']
                label = data[sent_id]['exercise value']
                tsv_writer.writerow([sent, label])
import json
import sys
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('-i','--id_file', help='id file', required=True)
parser.add_argument('-data', default='models/data.pt', help='data file')
parser.add_argument('-dest', help='destination path', required=True)

args = parser.parse_args()

id_set = set([])
counter = 0
with open(args.id_file) as fh:
    for line in fh:
        id_set.add(line.strip())
print('# ids: {}'.format(len(id_set)))
data = torch.load(args.data)
n = data['settings']['n_instances']
new_data = {}
new_data['dict'] = data['dict']
new_data['settings'] = data['settings']
new_data['src_embeddings'] = data['src_embeddings']
new_data['src'] = []
new_data['tgt'] = []
new_data['tgt_nums'] = []
new_data['ans'] = []
new_data['numbers'] = []
new_data['idx2id'] = {}

l = 0
for i in range(n):
    if data['idx2id'][i] in id_set:
        id_set.remove(data['idx2id'][i])
        new_data['src'].append(data['src'][i])
        new_data['tgt'].append(data['tgt'][i])
        new_data['tgt_nums'].append(data['tgt_nums'][i])
        new_data['ans'].append(data['ans'][i])
        new_data['numbers'].append(data['numbers'][i])
        new_data['idx2id'][l] = data['idx2id'][i]
        l += 1
new_data['settings']['n_instances'] = l
print(new_data['settings'])
print("unused ids:", id_set, len(id_set))

torch.save(new_data, args.dest)

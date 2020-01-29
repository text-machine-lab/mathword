import json
import sys
import os
from collections import OrderedDict

right_answers_path = sys.argv[1]
right_answers = os.path.join(right_answers_path, 'right_answers.json')
# right_answers = os.path.join(right_answers_path, 'overcome.json')
print("found right answers:", right_answers)

with open("/data2/ymeng/dolphin18k/eval_dataset/eval_dataset_shuffledv2.json") as f:
    source = json.load(f)
source_data = OrderedDict()

for item in source:
    source_data[item['id']] = item


with open(right_answers) as f:
    new_data = json.load(f)

for item in new_data:
    yahoo_id = item['id']
    source_data[yahoo_id]['equations'] = item['chosen_equations']

with open("/data2/ymeng/dolphin18k/eval_dataset/eval_dataset_auto_corrected.json", "w") as f:
    json.dump([source_data[k] for k in source_data], f, indent=2)
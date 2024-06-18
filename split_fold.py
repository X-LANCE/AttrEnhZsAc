from pathlib import Path

import pandas as pd
import json
import numpy as np


VGGSOUND_PATH = "./data/vggsound"
VGGSOUND_PATH = Path(VGGSOUND_PATH)

MAPPING_FILE = VGGSOUND_PATH / 'features/vggsound_mapping.json'
CATEGORY_FILE = VGGSOUND_PATH / 'features/stat.csv'
INPUT_FILE = VGGSOUND_PATH / 'features/vggsound_train_index_classname.hdf5'

with open(MAPPING_FILE, 'r') as input:
    class2int = json.load(input)
    # class_name -> integer target

# split category-wise
# OUTPUT_FILE = 'category_wise.json'
# df = pd.read_csv(CATEGORY_FILE, sep=',').set_index("class")
# class2category = df.category.to_dict()
# output_json = {}
# categrory2classes = {}
# for _class, _category in class2category.items():
#     categrory2classes.setdefault(_category, []).append(_class)
#     output_json.setdefault('all_classes', []).append(_class)
# output_json['folds'] = categrory2classes
# with open(OUTPUT_FILE, 'w') as output:
#     json.dump(output_json, output)

# randomly split all classes
OUTPUT_FILE = 'class_wise.json'
NUM_FOLDS = 5
import h5py
with h5py.File(INPUT_FILE, 'r') as input:
    labels = input['label'][:]
labels = list(map(lambda x: x.decode(), labels))
class_counter = {}
for _label in labels:
    class_counter[_label] = class_counter.setdefault(_label, 0) + 1
label_nums = list(class_counter.items())
label_nums.sort(key=lambda x: x[1], reverse=True)

folds = {}
# seq = 0
# for label, num in label_nums:
#     folds.setdefault(f'fold{seq}', []).append([label, num])
#     seq = (seq + 1) % NUM_FOLDS
random_state = np.random.RandomState(seed=0)
seqs = list(range(NUM_FOLDS))
random_state.shuffle(seqs)
for label, _ in label_nums:
    if not seqs:
        seqs = list(range(NUM_FOLDS))
        random_state.shuffle(seqs)
    seq = seqs.pop()
    folds.setdefault(f'fold{seq + 1}', []).append(label)
output_json = {
    'all_classes': list(class_counter.keys()),
    'folds': folds
}
with open(OUTPUT_FILE, 'w') as output:
    json.dump(output_json, output)
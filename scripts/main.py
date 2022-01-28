#from summary_dataset import SummaryDataset
import json
import os
import os.path as op


DATA_DIR = "DialogSum_Data"

SAMPLE_DATA = op.join(DATA_DIR, 'dialogsum.sample.jsonl')
TEST_DATA = op.join(DATA_DIR, 'dialogsum.test.jsonl')
TRAIN_DATA = op.join(DATA_DIR, 'dialogsum.train.jsonl')
DEV_DATA = op.join(DATA_DIR, 'dialogsum.dev.jsonl')

def load_jsonl(filepath):
    
    output_list = []
    
    with open(filepath) as sd_file:
        lines = sd_file.readlines()
    
    for line in lines:
        output_list.append(json.loads(line))

    return output_list

print(load_jsonl(TRAIN_DATA))

import pdb; pdb.set_trace()



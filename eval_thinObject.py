import matplotlib.pyplot as plt

import sys
import numpy as np
import torch

sys.path.insert(0, '..')
from isegm.utils import vis, exp

from isegm.inference import utils
from isegm.inference.evaluation import evaluate_dataset, evaluate_sample, evaluate_thin_object_5k
from isegm.inference.predictors import get_predictor


device = torch.device('cuda:1')
cfg = exp.load_config_file('config.yml', return_edict=True)

DATASET = 'ThinObject5k'

dataset = utils.get_dataset(DATASET, cfg)

print(len(dataset))

EVAL_MAX_CLICKS = 4
MODEL_THRESH = 0.49

checkpoint_path = utils.find_checkpoint(cfg.INTERACTIVE_MODELS_PATH, 'last_checkpoint')
model = utils.load_is_model(checkpoint_path, device)

# Possible choices: 'NoBRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C', 'RGB-BRS', 'DistMap-BRS'
brs_mode = 'f-BRS-B'
predictor = get_predictor(model, brs_mode, device, prob_thresh=MODEL_THRESH)

TARGET_IOU = 0.9
start_index = 0
all_ious, elapsed_time = evaluate_thin_object_5k(dataset, predictor, start_index, pred_thr=0.49,
                                        max_clicks=EVAL_MAX_CLICKS)
all_ious = np.array(all_ious)
print(all_ious.shape)
print(elapsed_time)
print(f"1 click: {all_ious[:, 0].mean()}")
print(f"2 clicks: {all_ious[:, 1].mean()}")
print(f"3 clicks: {all_ious[:, 2].mean()}")
print(f"4 clicks: {all_ious[:, 3].mean()}")
fname = "all_ious_last.npy"
np.save(fname, np.array(all_ious))
print(f"Saved to {fname}")

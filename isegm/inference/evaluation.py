from time import time

import numpy as np
import torch

from isegm.inference import utils
from isegm.inference.clicker import Clicker

try:
    get_ipython()
    from tqdm import tqdm_notebook as tqdm
except NameError:
    from tqdm import tqdm


def evaluate_dataset(dataset, predictor, **kwargs):
    all_ious = []

    start_time = time()
    for index in tqdm(range(len(dataset)), leave=False):
        sample = dataset.get_sample(index)

        _, sample_ious, _ = evaluate_sample(sample.image, sample.gt_mask, predictor,
                                            sample_id=index, **kwargs)
        all_ious.append(sample_ious)
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time


def evaluate_sample(image, gt_mask, predictor, max_iou_thr,
                    pred_thr=0.49, min_clicks=1, max_clicks=20,
                    sample_id=None, callback=None):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []

    with torch.no_grad():
        predictor.set_input_image(image)

        for click_indx in range(max_clicks):
            clicker.make_next_click(pred_mask)
            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr

            if callback is not None:
                callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

            iou = utils.get_iou(gt_mask, pred_mask)
            ious_list.append(iou)

            if iou >= max_iou_thr and click_indx + 1 >= min_clicks:
                break

        return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs

def evaluate_thin_object_5k_sample(image, gt_mask, predictor,
                    pred_thr=0.49, min_clicks=1, max_clicks=4,
                    sample_id=None, callback=None):
    clicker = Clicker(gt_mask=gt_mask)
    pred_mask = np.zeros_like(gt_mask)
    ious_list = []
    # max_clicks = 4
    with torch.no_grad():
        predictor.set_input_image(image)
        for click_indx in range(max_clicks):
            clicker.make_next_click(pred_mask)
            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr

            if callback is not None:
                callback(image, gt_mask, pred_probs, sample_id, click_indx, clicker.clicks_list)

            iou = utils.get_iou(gt_mask, pred_mask)
            ious_list.append(iou)
    return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs

def evaluate_thin_object_5k(dataset, predictor, start_index, **kwargs):
    all_ious = []

    start_time = time()
    for index in tqdm(range(start_index, len(dataset)), leave=False):
        #print(index, flush=True)
        sample = dataset.get_sample(index)
        try:
            _, sample_ious, _ = evaluate_thin_object_5k_sample(sample.image, sample.gt_mask, predictor,
                                            sample_id=index, **kwargs)
        except RuntimeError:
            print("CUDA out of memory. Finished on [", index, "]:", dataset.dataset_samples[index])
            fname = "all_ious" + str(start_index) + "_" + str(index) + ".npy"
            np.save(fname, np.array(all_ious))
            print(f"Saved to {fname}")
            exit()
        all_ious.append(sample_ious)
        
    end_time = time()
    elapsed_time = end_time - start_time

    return all_ious, elapsed_time

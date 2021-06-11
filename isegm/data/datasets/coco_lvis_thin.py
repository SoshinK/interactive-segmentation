from pathlib import Path
import pickle
import random
import numpy as np
import json
import cv2
from copy import Error, deepcopy
from isegm.data.base import ISDataset
from isegm.data.sample import DSample

import matplotlib.pyplot as plt

class CocoLvisThinObject5kDataset(ISDataset):
    def __init__(self, cocolvis_dataset_path, thinobject5k_dataset_path, split='train', stuff_prob=0.0,
                 cocolvis_allow_list_name=None, thinobject5k_allow_list_name=None, 
                 cocolvis_anno_file='hannotation.pickle', thinobject5k_anno_file='train_instances.pkl', cocolvis_size=None, **kwargs):
        super(CocoLvisThinObject5kDataset, self).__init__(**kwargs)
        cocolvis_dataset_path = Path(cocolvis_dataset_path)
        self._cocolvis_split_path = cocolvis_dataset_path / split
        self.split = split
        self._cocolvis_images_path = self._cocolvis_split_path / 'images'
        self._cocolvis_masks_path = self._cocolvis_split_path / 'masks'
        self.stuff_prob = stuff_prob

        with open(self._cocolvis_split_path / cocolvis_anno_file, 'rb') as f:
            self._cocolvis_dataset_samples = sorted(pickle.load(f).items())

        if cocolvis_allow_list_name is not None:
            cocolvis_allow_list_path = self._cocolvis_split_path / cocolvis_allow_list_name
            with open(cocolvis_allow_list_path, 'r') as f:
                allow_images_ids = json.load(f)
            allow_images_ids = set(allow_images_ids)

            self._cocolvis_dataset_samples = [sample for sample in self._cocolvis_dataset_samples
                                    if sample[0] in allow_images_ids]


        thinobject5k_dataset_path = Path(thinobject5k_dataset_path)
        self._thinobject5k_split_path = thinobject5k_dataset_path
        self._thinobject5k_images_path = self._thinobject5k_split_path / 'images'
        self._thinobject5k_images_masks = self._thinobject5k_split_path / 'masks'
        with open(self._thinobject5k_split_path / thinobject5k_anno_file, 'rb') as f:
            self._thinobject5k_dataset_samples = sorted(pickle.load(f).items())
        if thinobject5k_allow_list_name is not None:
            thinobject5k_allow_list_path = self._thinobject5k_split_path / thinobject5k_allow_list_name
            with open(thinobject5k_allow_list_path, 'r') as f:
                allow_images_ids = json.load(f)
            allow_images_ids = set(allow_images_ids)

            self._thinobject5k_dataset_samples = [sample for sample in self._thinobject5k_dataset_samples
                                    if sample[0] in allow_images_ids]

        if cocolvis_size is None:
            self.dataset_samples = np.concatenate((self._thinobject5k_dataset_samples, self._cocolvis_dataset_samples))
        elif cocolvis_size < 0 or cocolvis_size > len(self._cocolvis_dataset_samples) / (len(self._cocolvis_dataset_samples) + len(self._thinobject5k_dataset_samples)):
            raise ValueError("Wrong cocolvis_size")
        else:
            add_samples_from_cocolvis = cocolvis_size / (1 - cocolvis_size) * len(self._thinobject5k_dataset_samples)
            self.dataset_samples = np.concatenate((
                self._thinobject5k_dataset_samples, 
                np.array(self._cocolvis_dataset_samples)[::(len(self._cocolvis_dataset_samples) // add_samples_from_cocolvis)])) #bullshit 
        print(">> ", len(self.dataset_samples), len(self._cocolvis_dataset_samples), len(self._cocolvis_dataset_samples) / len(self.dataset_samples), cocolvis_size)
        print("??", self.dataset_samples[0], self._cocolvis_dataset_samples[0])

    def get_sample(self, index) -> DSample:
        image_id, sample = self.dataset_samples[index]
        # print(image_id)
        if (image_id, sample) in self._cocolvis_dataset_samples:
            image_path = self._cocolvis_images_path / f'{image_id}.jpg'
            packed_masks_path = self._cocolvis_masks_path / f'{image_id}.pickle'
            
            with open(packed_masks_path, 'rb') as f:
                encoded_layers, objs_mapping = pickle.load(f)
            layers = [cv2.imdecode(x, cv2.IMREAD_UNCHANGED) for x in encoded_layers]
            layers = np.stack(layers, axis=2)

            instances_info = deepcopy(sample['hierarchy'])
            for inst_id, inst_info in list(instances_info.items()):
                if inst_info is None:
                    inst_info = {'children': [], 'parent': None, 'node_level': 0}
                    instances_info[inst_id] = inst_info
                inst_info['mapping'] = objs_mapping[inst_id]

            if self.stuff_prob > 0 and random.random() < self.stuff_prob:
                for inst_id in range(sample['num_instance_masks'], len(objs_mapping)):
                    instances_info[inst_id] = {
                        'mapping': objs_mapping[inst_id],
                        'parent': None,
                        'children': []
                    }
            else:
                for inst_id in range(sample['num_instance_masks'], len(objs_mapping)):
                    layer_indx, mask_id = objs_mapping[inst_id]
                    layers[:, :, layer_indx][layers[:, :, layer_indx] == mask_id] = 0

            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            return DSample(image, layers, objects=instances_info)

        elif (image_id, sample) in self._thinobject5k_dataset_samples:
            image_path = self._thinobject5k_images_path / ((f'{image_id}')[:-3] + 'jpg')
            packed_masks_path = self._thinobject5k_images_masks / f'{image_id}'

            layers = cv2.imread(str(packed_masks_path))[:, :, 0].astype(np.int32)
            layers[layers < 128] = 0
            layers[layers > 128] = 1

            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            return DSample(image, layers, objects_ids=[1])
        else:
            raise Error("Something wrong...")
    
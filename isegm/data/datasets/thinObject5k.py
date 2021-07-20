from pathlib import Path
from copy import Error, deepcopy
import pickle
import json
import cv2
import numpy as np

from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class ThinObject5k_TESTONLY(ISDataset):
    def __init__(self, dataset_path, split='test',
                 images_dir_name='images', masks_dir_name='masks',
                 **kwargs):
        super(ThinObject5k, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)
        assert self.dataset_path.exists()
        self._images_path = self.dataset_path / images_dir_name
        self._masks_paths = self.dataset_path / masks_dir_name

        self.dataset_samples = [Path(img_name[:-1]).stem for img_name in open(self.dataset_path / ('list/' + split + '.txt')).readlines()]

    

    def get_sample(self, index) -> DSample:
        image_name = self.dataset_samples[index]
        image_path = str(self._images_path / image_name) + '.jpg'
        mask_path = str(self._masks_paths / image_name) + '.png'

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = cv2.imread(mask_path)[:, :, 0].astype(np.int32)
        instances_mask[instances_mask < 128] = 0
        instances_mask[instances_mask > 128] = 1

        return DSample(image, instances_mask, objects_ids=[1], sample_id=index)

class ThinObject5k(ISDataset):
    def __init__(self, thinobject5k_dataset_path, split='train', stuff_prob=0.0, **kwargs):
        super(ThinObject5k, self).__init__(**kwargs)
        # cocolvis_dataset_path = Path(cocolvis_dataset_path)
        # self._cocolvis_split_path = cocolvis_dataset_path / split
        self.split = split
        # self._cocolvis_images_path = self._cocolvis_split_path / 'images'
        # self._cocolvis_masks_path = self._cocolvis_split_path / 'masks'
        self.stuff_prob = stuff_prob

        # with open(self._cocolvis_split_path / cocolvis_anno_file, 'rb') as f:
        #     self._cocolvis_dataset_samples = sorted(pickle.load(f).items())

        # if cocolvis_allow_list_name is not None:
        #     cocolvis_allow_list_path = self._cocolvis_split_path / cocolvis_allow_list_name
        #     with open(cocolvis_allow_list_path, 'r') as f:
        #         allow_images_ids = json.load(f)
        #     allow_images_ids = set(allow_images_ids)

        #     self._cocolvis_dataset_samples = [sample for sample in self._cocolvis_dataset_samples
        #                             if sample[0] in allow_images_ids]


        thinobject5k_dataset_path = Path(thinobject5k_dataset_path)
        # self._thinobject5k_split_path = thinobject5k_dataset_path
        self._thinobject5k_images_path = thinobject5k_dataset_path / 'images'
        self._thinobject5k_images_masks = thinobject5k_dataset_path / 'masks'
        with open(thinobject5k_dataset_path / (self.split + '_instances.pkl'), 'rb') as f:
            self.dataset_samples = sorted(pickle.load(f).items())
        
        # if cocolvis_size is None:
        #     self.dataset_samples = np.concatenate((self._thinobject5k_dataset_samples, self._cocolvis_dataset_samples))
        # elif cocolvis_size < 0 or cocolvis_size > len(self._cocolvis_dataset_samples) / (len(self._cocolvis_dataset_samples) + len(self._thinobject5k_dataset_samples)):
        #     raise ValueError("Wrong cocolvis_size")
        # else:
        #     add_samples_from_cocolvis = cocolvis_size / (1 - cocolvis_size) * len(self._thinobject5k_dataset_samples)
        #     print(add_samples_from_cocolvis)
        #     self.dataset_samples = np.concatenate((
        #         self._thinobject5k_dataset_samples, 
        #         np.array(self._cocolvis_dataset_samples)[::(len(self._cocolvis_dataset_samples) // int(add_samples_from_cocolvis))])) #bullshit 
        # print(">> ", len(self.dataset_samples), len(self._cocolvis_dataset_samples), len(self._cocolvis_dataset_samples) / len(self.dataset_samples), cocolvis_size)
        # print("??", self.dataset_samples[0], self._cocolvis_dataset_samples[0])

    def get_sample(self, index) -> DSample:
        image_id, sample = self.dataset_samples[index]
        # print(image_id)

        image_path = self._thinobject5k_images_path / ((f'{image_id}')[:-3] + 'jpg')
        packed_masks_path = self._thinobject5k_images_masks / f'{image_id}'

        layers = cv2.imread(str(packed_masks_path))[:, :, 0].astype(np.int32)
        layers[layers < 128] = 0
        layers[layers > 128] = 1

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return DSample(image, layers, objects_ids=[1])


def test():
    pass

if __name__ == '__main__':
    test()

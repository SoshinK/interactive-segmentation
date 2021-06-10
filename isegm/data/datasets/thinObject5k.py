from pathlib import Path

import cv2
import numpy as np

from isegm.data.base import ISDataset
from isegm.data.sample import DSample


class ThinObject5k(ISDataset):
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

def test():
    pass

if __name__ == '__main__':
    test()

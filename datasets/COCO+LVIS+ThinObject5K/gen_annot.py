import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
# f = open('datasets/ThinObject5K/trainval_instances.pkl', 'rb')
# dataset_samples_thin = sorted(pickle.load(f).items())

# print(dataset_samples_thin[0])
# print(dataset_samples_thin[1])

# f = open('datasets/COCO+LVIS/train/hannotation.pickle', 'rb')
# dataset_samples_cocolvis = sorted(pickle.load(f).items())

# print(dataset_samples_cocolvis[0])
# print(dataset_samples_cocolvis[1])

# f = open('datasets/COCO+LVIS/train/masks/000000000030.pickle', 'rb')

# encoded_layers, objs_mapping = pickle.load(f)
# layers = [cv2.imdecode(x, cv2.IMREAD_UNCHANGED) for x in encoded_layers]
# layers = np.stack(layers, axis=2)
# print(layers.shape)
# # plt.imshow(layers)
# # plt.show()

# print(objs_mapping)

# sample = dataset_samples_cocolvis[0][1]
# print(np.unique(layers))

# for inst_id in range(sample['num_instance_masks'], len(objs_mapping)):
#     layer_indx, mask_id = objs_mapping[inst_id]
#     layers[:, :, layer_indx][layers[:, :, layer_indx] == mask_id] = 0

# # # plt.imshow(layers * 50)
# # plt.show()

# print(np.unique(layers))



f = open('datasets/ThinObject5K/trainval_instances.pkl', 'rb')
trainval = sorted(pickle.load(f).keys())
f = open('datasets/ThinObject5K/train_instances.pkl', 'rb')
train = sorted(pickle.load(f).keys())
# print(train)
val = list(set(trainval) - set(train))
val = {v:{0: 1} for v in val}

with open('datasets/ThinObject5K/val_instances.pkl', 'wb') as f:
    pickle.dump(val, f)

print(len(trainval), len(train))
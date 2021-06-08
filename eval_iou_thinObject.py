import numpy as np

a1 = np.load("all_ious0_110.npy")
print(a1.shape)
a2 = np.load("all_ious110_304.npy")
print(a2.shape)
a3 = np.load("all_ious305_351.npy")
print(a3.shape)
a4 = np.load("all_ious352_404.npy")
print(a4.shape)
a5 = np.load("all_ious405_410.npy")
print(a5.shape)
a6 = np.load("all_ious_last.npy")
print(a6.shape)

a = np.concatenate((a1, a2, a3, a4, a5, a6))
print(a.shape)
print("1 click : ", round(a1[:, 0].mean(), 2))
print("2 clicks : ", round(a1[:, 1].mean(), 2))
print("3 clicks : ", round(a1[:, 2].mean(), 2))
print("4 clicks : ", round(a1[:, 3].mean(), 2))


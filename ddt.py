import numpy as np

data = np.load("/Users/tunm/datasets/ballFaceDataset20230317-FMESH/1654053172386_whiteLeft_left/1654053172386_whiteLeft_left.npy", allow_pickle=True)

data = dict(data[()])

print(data['verts'])
import os

import cv2
import numpy as np

def get_aligned_lmk(kps, mat):

    ones = np.ones((kps.shape[0], 1))

    lmk_h = np.concatenate((kps, ones), axis=1)

    align_points = np.dot(mat, lmk_h.T).T

    return align_points

def trans_to_align(image, kps, input_size=256):
    from skimage import transform as trans
    new_size = 144
    dst_pts = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]], dtype=np.float32)
    dst_pts[:, 0] += ((new_size - 112) // 2)
    dst_pts[:, 1] += 8
    dst_pts[:, :] *= (input_size / float(new_size))
    tf = trans.SimilarityTransform()
    tf.estimate(kps, dst_pts)
    tform = tf.params[0:2, :]
    warped = cv2.warpAffine(image, tform, (input_size,) * 2)

    return tform, warped

input_size = 256
item_data = "/Users/tunm/datasets/ballFaceDataset20230317-FMESH/1654053172386_whiteLeft_left"
raw_image_path = os.path.sep.join([item_data, 'raw_image.jpg'])
trans_matrix_path = os.path.sep.join([item_data, 'to256_trans_matrix.npy'])
mesh_path = os.path.sep.join([item_data, 'raw_mesh.npy'])
kps5_path = os.path.sep.join([item_data, 'kps5.npy'])

trans_matrix = np.load(trans_matrix_path)
mesh = np.load(mesh_path)
kps5 = np.load(kps5_path)

raw_image = cv2.imread(raw_image_path)

warped = cv2.warpAffine(raw_image, trans_matrix, (input_size, ) * 2, )
warped_mesh = get_aligned_lmk(mesh, trans_matrix)
warped_kps5 = get_aligned_lmk(kps5, trans_matrix)

for x, y in warped_mesh.astype(int):
    cv2.circle(warped, (x, y), 0, (0, 0, 255), 1)

for x, y in warped_kps5.astype(int):
    cv2.circle(warped, (x, y), 0, (0, 255, 0), 3)

cv2.imshow("w", warped)
cv2.waitKey(0)

print(mesh.shape)


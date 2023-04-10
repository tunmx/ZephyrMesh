import os

import cv2
import numpy as np

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

item_data = "/Users/tunm/datasets/ballFaceDataset20230317-FMESH/1654053172386_whiteLeft_left"

raw_image_path = os.path.sep.join([item_data, 'raw_image.jpg'])
kps68_path = os.path.sep.join([item_data, 'kps68.npy'])
mesh_path = os.path.sep.join([item_data, 'raw_mesh.npy'])

raw_image = cv2.imread(raw_image_path)
kps68 = np.load(kps68_path)
mesh = np.load(mesh_path)

for x, y in mesh.astype(int):
    cv2.circle(raw_image, (x, y), 0, (0, 0, 255), 5)

for x, y in kps68.astype(int):
    cv2.circle(raw_image, (x, y), 0, (0, 255, 0), 10)

cv2.imshow("w", raw_image)
cv2.waitKey(0)


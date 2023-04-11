import cv2
import numpy as np

image = cv2.imread("/Users/tunm/datasets/ballFaceDataset20230317-FMESH/1654049770649_whiteFace_face/raw_image.jpg")
kps = np.load("/Users/tunm/datasets/ballFaceDataset20230317-FMESH/1654049770649_whiteFace_face/raw_mesh.npy")

black = np.ones_like(image, dtype=np.uint8) * 255
for idx, (x, y) in enumerate(kps.astype(int)):
    cv2.circle(black, (x, y), 0, (0, 255, 0), 3)
    cv2.putText(black, str(idx), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 80), 1)

cv2.imwrite("black.jpg", black)

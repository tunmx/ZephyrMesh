import time

import cv2
import numpy as np
import data.fmesh.image_tools as its
from fd_tools.scrfd import SCRFD
import onnxruntime as ort

session = ort.InferenceSession("workplace/s2/best_model.onnx", None)
input_config = session.get_inputs()[0]
output_config = session.get_outputs()[0]

det = SCRFD("resource/scrfd_2.5g_bnkps_shape320x320.onnx")

cam = cv2.VideoCapture(0)

while True:
    ret, image = cam.read()
    bboxes, kpss = det.detect(image)
    input_size = 256
    if kpss.size > 0:
        kps = kpss[0]
        tform, warped = its.trans_to_align(image, kps, input_size=256)

        data, _ = its.data_normalization(warped)
        t = time.time()
        result = session.run([output_config.name], {input_config.name: [data]})
        print("cost:", time.time() - t)
        mesh = result[0].reshape(-1, 2) * input_size
        print(mesh)

        ones = np.ones((mesh.shape[0], 1))

        mesh_c = np.concatenate((mesh, ones), axis=1)
        inv_matrix = cv2.invertAffineTransform(tform)
        mesh_t = np.dot(inv_matrix, mesh_c.T).T
        print(mesh_t)
        for x, y in mesh_t.astype(int):
            cv2.circle(image, (x, y), 0, (0, 255, 255), 2)

    cv2.imshow('w', image)
    cv2.waitKey(1)
import cv2
import numpy as np
import glob
import os


def images_to_square(images, pad=4, resize_pad=None):
    n, h, w, c = images.shape
    assert n == pad * pad
    new_list = list()
    for idx, img in enumerate(images):
        if resize_pad:
            img = cv2.resize(img, (resize_pad, resize_pad))
        new_list.append(img)

    array_images = np.asarray(new_list).reshape((4, 4, h, w, c))
    cols = list()
    for rows in array_images:
        lines = np.concatenate(rows, axis=1)
        cols.append(lines)
    square = np.concatenate(cols, axis=0)

    return square


def encode_images(image: np.ndarray):
    image_encode = image / 255.0
    if len(image_encode.shape) == 4:
        image_encode = image_encode.transpose(0, 3, 1, 2)
    else:
        image_encode = image_encode.transpose(2, 0, 1)
    image_encode = image_encode.astype(np.float32)

    return image_encode


def decode_images(images_tensor):
    image_decode = images_tensor.detach().numpy() * 255
    image_decode = image_decode.transpose(0, 2, 3, 1)
    image_decode = image_decode.astype(np.uint8)
    image_decode = image_decode.copy()

    return image_decode


def decode_points(label_tensor, w, h):
    kps = label_tensor.detach().numpy().reshape(-1, 1220, 2)
    kps[:, :, 0] *= w
    kps[:, :, 1] *= h

    return kps


colors = [(100, 100, 255), (10, 255, 100), (100, 190, 240), (100, 255, 255)]


def visual_images(images_tensor, label_tensor, w, h, swap=True, color=(96, 48, 255)):
    images = decode_images(images_tensor)
    kps = decode_points(label_tensor, w, h)
    list_ = list()
    for idx, img in enumerate(images):
        if swap:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i, (x, y) in enumerate(kps[idx].astype(np.int32)):
            cv2.line(img, (x, y), (x, y), color, 1)

        list_.append(img)

    return list_


def get_aligned_lmk(kps, mat):
    ones = np.ones((kps.shape[0], 1))

    lmk_h = np.concatenate((kps, ones), axis=1)

    align_points = np.dot(mat, lmk_h.T).T

    return align_points


def trans_to_align(image, kps, input_size=256, center_disturbance=0.0):
    from skimage import transform as trans
    import numpy as np
    import cv2

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

    center_disturb = np.random.uniform(-center_disturbance, center_disturbance, size=(2,))
    tform[:, 2] += center_disturb  # Add the center disturbance to the translation part of the transform matrix

    warped = cv2.warpAffine(image, tform, (input_size,) * 2)

    return tform, warped



def kps_flip(width, kps):
    n_kps = kps.copy()
    n_kps[:, 0] = width - n_kps[:, 0] - 1
    # n_kps = n_kps[n_kps[:, 0].argsort()]
    return n_kps


def find_image_file(file_path):
    """
    给定文件路径和文件名（不带后缀），自动匹配后缀名为 png、jpg、jpeg 的图像文件，并返回文件名及其完整路径。
    如果找不到匹配的文件，则返回 None。
    """
    extensions = ['png', 'jpg', 'jpeg']
    for ext in extensions:
        pattern = os.path.join(file_path + '.' + ext)
        files = glob.glob(pattern)
        if len(files) > 0:
            return files[0]
    return None


def data_normalization(image: np.ndarray, points: np.ndarray = None, ):
    height, width, _ = image.shape
    data = image / 255.0
    data = data.transpose(2, 0, 1)
    label = None
    if points is not None:
        label = points.copy()
        label[:, 0] /= width
        label[:, 1] /= height
        label = label.reshape(-1).astype(np.float32)

    return data.astype(np.float32), label

import glob
import os
import click
import numpy as np
import cv2
import tqdm
from loguru import logger


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


def _load_data(data_dir):
    candidate = [os.path.sep.join([data_dir, item]) for item in os.listdir(data_dir) if
                 os.path.isdir(os.path.sep.join([data_dir, item]))]
    selected = list()
    for idx, sub_dir in enumerate(tqdm.tqdm(candidate)):
        basename = os.path.basename(sub_dir)
        kps5_path = os.path.sep.join([sub_dir, "kps5.npy"])
        mesh_path = os.path.sep.join([sub_dir, "raw_mesh.npy"])
        trans_matrix_path = os.path.sep.join([sub_dir, "to256_trans_matrix.npy"])
        raw_image_path = os.path.sep.join([sub_dir, "raw_image"])
        raw_image_path = find_image_file(raw_image_path)
        save_map = list()
        for path in [kps5_path, mesh_path, trans_matrix_path, raw_image_path]:
            is_save = True
            if not os.path.exists(path) or path is None:
                logger.warning(f"[Not Found Data]{basename}")
                is_save = False
            save_map.append(is_save)
        if all(save_map):
            mate = dict(basename=basename, kps5=kps5_path, mesh=mesh_path, trans_matrix=trans_matrix_path,
                        raw_image=raw_image_path)
            selected.append(mate)

    return selected


image_size = 256


@click.command()
@click.option('-src', '--src', type=click.Path(exists=True))
@click.option('-save', '--save', type=click.Path())
def generate(src, save):
    data = _load_data(src)
    for mate in tqdm.tqdm(data):
        folder = os.path.sep.join([save, mate['basename']])
        os.makedirs(folder, exist_ok=True)
        kp5 = np.load(mate['kps5'])
        mesh = np.load(mate['mesh'])
        trans_matrix = np.load(mate['trans_matrix'])
        raw_image = cv2.imdecode(np.fromfile(mate['raw_image'], dtype=np.uint8), cv2.IMREAD_COLOR)
        warped = cv2.warpAffine(raw_image, trans_matrix, (image_size,) * 2, )
        warped_mesh = get_aligned_lmk(mesh, trans_matrix)
        warped_lmk5 = get_aligned_lmk(kp5, trans_matrix)
        cv2.imwrite(os.path.sep.join([folder, "image.png"]), warped)
        np.save(os.path.sep.join([folder, "kp5.npy"]), warped_lmk5)
        np.save(os.path.sep.join([folder, "mesh.npy"]), warped_mesh)


if __name__ == '__main__':
    generate()

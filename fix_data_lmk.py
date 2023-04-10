import os
import numpy as np
import json
import click
import tqdm

n68 = [0, 1, 73, 2, 3, 4, 76, 5, 6, 7, 79, 8, 9, 10, 82, 11, 12, 84, 23, 24, 25, 85, 86, 40, 41, 42, 87, 108, 109, 110,
       57, 50, 51, 115, 52, 53, 13, 14, 16, 17, 18, 20, 30, 31, 33, 34, 35, 37, 58, 59, 120, 60, 123, 61, 62, 63, 129,
       64, 130, 65, 116, 136, 67, 139, 117, 144, 70, 147]

@click.command()
@click.argument("raw_images_path")
@click.argument("fitting_dataset_path")
def run(raw_images_path, fitting_dataset_path):

    suffix = "json"

    dirs_list = [os.path.join(fitting_dataset_path, item) for item in os.listdir(fitting_dataset_path) if os.path.isdir(os.path.join(fitting_dataset_path, item))]

    for item in tqdm.tqdm(dirs_list):
        basename = os.path.basename(item)
        json_file = os.path.join(raw_images_path, basename + f".{suffix}")
        with open(json_file, "r") as f:
            content = json.load(f)
        result = content['result']
        face = result['face_list'][0]
        landmark201 = face['landmark201']
        kps201 = np.asarray([[v['x'], v['y']] for k, v in landmark201.items()])
        kps68 = kps201[n68]

        kps68_path = os.path.join(fitting_dataset_path, basename, 'kps68.npy')
        kps201_path = os.path.join(fitting_dataset_path, basename, 'kps201.npy')

        np.save(kps68_path, kps68)
        np.save(kps201_path, kps201)


if __name__ == '__main__':
    run()
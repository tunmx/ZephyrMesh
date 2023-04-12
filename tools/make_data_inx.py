import random

import click
import os
import numpy as np

def write_to_file(filename, data):
    with open(filename, 'w') as f:
        for item in data:
            f.write("%s\n" % item)

@click.command()
@click.option('-src', '--src', type=click.Path(exists=True))
@click.option('-save', '--save', type=click.Path())
@click.option('-rate', nargs=3, type=float)
def generate(src, save, rate):
    os.makedirs(save, exist_ok=True)
    assert np.isclose(sum(rate), 1.0, rtol=1e-05, atol=1e-08, equal_nan=False), "The rates must sum to 1.0"
    candidate = [os.path.sep.join([src, item]) for item in os.listdir(src) if
                 os.path.isdir(os.path.sep.join([src, item]))]
    selected = list()
    error_data = 0
    for idx, sub_dir in enumerate(candidate):
        basename = os.path.basename(sub_dir)
        kps5_path = os.path.sep.join([sub_dir, "kp5.npy"])
        mesh_path = os.path.sep.join([sub_dir, "mesh.npy"])
        raw_image = os.path.sep.join([sub_dir, "image.png"])
        for path in [kps5_path, mesh_path, raw_image]:
            if not os.path.exists(path):
                print(path, "is not found")
                error_data += 1
                continue
        selected.append(basename)

    random.shuffle(selected)
    n = len(selected)
    train_size = int(n * rate[0])
    val_size = int(n * rate[1])
    test_size = n - train_size - val_size

    train_data = selected[:train_size]
    val_data = selected[train_size:train_size + val_size]
    test_data = selected[-test_size:]

    print("Train data:", len(train_data))
    print("Val data:", len(val_data))
    print("Test data:", len(test_data))
    print(f"has error data : {error_data}")

    write_to_file(os.path.sep.join([save, "train.txt"]), train_data)
    write_to_file(os.path.sep.join([save, "val.txt"]), val_data)
    write_to_file(os.path.sep.join([save, "test.txt"]), test_data)


if __name__ == '__main__':
    generate()
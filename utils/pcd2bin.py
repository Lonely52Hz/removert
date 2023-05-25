# https://github.com/Yuseung-Na/pcd2bin

import numpy as np
import os
import argparse
import pypcd
import csv
from tqdm import tqdm


def load_labels(label_path):
    """ Load semantic and instance labels in SemanticKitti format.
    """
    label = np.fromfile(label_path, dtype=np.uint32)
    label = label.reshape((-1))

    sem_label = label & 0xFFFF  # semantic label in lower half
    inst_label = label >> 16  # instance id in upper half

    # sanity check
    assert ((sem_label + (inst_label << 16) == label).all())

    return sem_label, inst_label


def main():
    # Add parser
    parser = argparse.ArgumentParser(description="Convert .pcd to .bin")
    parser.add_argument(
        "--pcd_path", '-p',
        help=".pcd file path.",
        type=str,
        default="/home/user/lidar_pcd"
    )
    parser.add_argument(
        "--bin_path", '-b',
        help=".bin file path.",
        type=str,
        default="/home/user/lidar_bin"
    )
    parser.add_argument(
        "--original_label_path",
        type=str,
        default="/home/user/lidar_bin"
    )
    parser.add_argument(
        "--new_label_path",
        type=str,
        default="/home/user/lidar_bin"
    )
    parser.add_argument(
        "--idx_path", '-i',
        type=str,
        default="/home/user/lidar_bin"
    )

    args = parser.parse_args()

    # Find all pcd files
    pcd_files = []
    for (path, dir, files) in os.walk(args.pcd_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.pcd':
                pcd_files.append(path + "/" + filename)

    # Sort pcd files by file name
    pcd_files.sort()
    print("Finish to load point clouds!")

    # Make bin_path directory
    try:
        if not (os.path.isdir(args.bin_path)):
            os.makedirs(os.path.join(args.bin_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise

    # Generate csv meta file
    csv_file_path = os.path.join(args.bin_path, "meta.csv")
    csv_file = open(csv_file_path, "w")
    meta_file = csv.writer(
        csv_file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL
    )
    # Write csv meta file header
    meta_file.writerow(
        [
            "pcd file name",
            "bin file name",
        ]
    )
    print("Finish to generate csv meta file")

    # Converting Process
    print("Converting Start!")
    for pcd_file in tqdm(pcd_files):
        # Get pcd file
        pc = pypcd.PointCloud.from_path(pcd_file)
        seq = pcd_file.split('/')[-1].split('.')[0]
        label_path = args.original_label_path + seq + '.label'
        sem_label, inst_label = load_labels(label_path)
        idx_path = args.idx_path + seq + '.txt'
        idx = list(np.loadtxt(idx_path,dtype=np.int32))
        sem_label = sem_label[idx]
        inst_label = inst_label[idx]
        new_label_path = os.path.join(args.new_label_path, seq + '.label')
        new_label = sem_label + (inst_label << 16)
        new_label.tofile(new_label_path)

        # Generate bin file name
        seq = int(seq)
        bin_file_name = "{:06d}.bin".format(seq)
        bin_file_path = os.path.join(args.bin_path, bin_file_name)

        # Get data from pcd (x, y, z, intensity, ring, time)
        np_x = (np.array(pc.pc_data['x'], dtype=np.float32)).astype(np.float32)
        np_y = (np.array(pc.pc_data['y'], dtype=np.float32)).astype(np.float32)
        np_z = (np.array(pc.pc_data['z'], dtype=np.float32)).astype(np.float32)
        np_i = (np.array(pc.pc_data['intensity'],
                dtype=np.float32)).astype(np.float32)
        # np_r = (np.array(pc.pc_data['ring'], dtype=np.float32)).astype(np.float32)
        # np_t = (np.array(pc.pc_data['time'], dtype=np.float32)).astype(np.float32)

        # Stack all data
        points_32 = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))

        # delete invalid points, e.g. depth == 0
        index = np.linalg.norm(points_32[:, 0:3], 2, axis=1) > 0.
        points_32 = points_32[index, :]

        # Save bin file
        points_32.tofile(bin_file_path)

        # Write csv meta file
        meta_file.writerow(
            [os.path.split(pcd_file)[-1], bin_file_name]
        )


    # delete the meta.csv file
    os.remove(csv_file_path)


if __name__ == "__main__":
    main()

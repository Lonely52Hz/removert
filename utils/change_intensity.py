# https://github.com/Yuseung-Na/pcd2bin

import numpy as np
import os
import argparse
import pypcd
import csv
from tqdm import tqdm
import open3d as o3d


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
        "--save_path", '-s',
        help="save file path.",
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

    # Make save_path directory
    try:
        if not (os.path.isdir(args.save_path)):
            os.makedirs(os.path.join(args.save_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise

    # Generate csv meta file
    csv_file_path = os.path.join(args.save_path, "meta.csv")
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

        # Generate bin file name
        bin_file_name = "{}.pcd".format(seq)
        bin_file_path = os.path.join(args.save_path, bin_file_name)

        # Get data from pcd (x, y, z, intensity, ring, time)
        np_x = (np.array(pc.pc_data['x'], dtype=np.float32)).astype(np.float32)
        np_y = (np.array(pc.pc_data['y'], dtype=np.float32)).astype(np.float32)
        np_z = (np.array(pc.pc_data['z'], dtype=np.float32)).astype(np.float32)
        np_i = pc.pc_data['intensity'].astype(dtype=np.uint32) & 0xFFFF
        # np_r = (np.array(pc.pc_data['ring'], dtype=np.float32)).astype(np.float32)
        # np_t = (np.array(pc.pc_data['time'], dtype=np.float32)).astype(np.float32)

        flag = (np_i == 252)

        for i in [253, 254, 255, 256, 257, 258, 259]:
            flag = np.logical_or(flag, np_i == i)
        
        np_i = np.where(flag, 0xFFFFFFFF, 0).astype(np.float32)

        # Stack all data    
        data = np.transpose(np.vstack((np_x, np_y, np_z, np_i)))   
        data = data.astype(np.float32)
        data_bytes = data.tobytes()

        with open(bin_file_path, 'w') as f:
            f.write('# .PCD v0.7 - Point Cloud Data file format\n')
            f.write('VERSION 0.7\n')
            f.write('FIELDS x y z intensity\n')
            f.write('SIZE 4 4 4 4\n')
            f.write('TYPE F F F F\n')
            f.write('COUNT 1 1 1 1\n')
            f.write('WIDTH %d\n' % data.shape[0])
            f.write('HEIGHT 1\n')
            f.write('POINTS %d\n' % data.shape[0])
            f.write('DATA binary\n')

            f.write(data_bytes)

        # Write csv meta file
        meta_file.writerow(
            [os.path.split(pcd_file)[-1], bin_file_name]
        )

    # delete the meta.csv file
    os.remove(csv_file_path)


if __name__ == "__main__":
    main()

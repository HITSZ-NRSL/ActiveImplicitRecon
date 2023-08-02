# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/integrate_scene.py

import argparse
from genericpath import exists, isfile
import json
from posixpath import join, splitext, lexists
import re
import numpy as np
import os, sys
import open3d as o3d

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyexample_path)


def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(file_list_ordered, key=alphanum_key)


def get_file_list(path, extension=None):
    if extension is None:
        file_list = [path + f for f in os.listdir(path) if isfile(join(path, f))]
    else:
        file_list = [
            path + f
            for f in os.listdir(path)
            if isfile(join(path, f)) and splitext(f)[1] == extension
        ]
    file_list = sorted_alphanum(file_list)
    return file_list


def add_if_exists(path_dataset, folder_names):
    for folder_name in folder_names:
        if lexists(join(path_dataset, folder_name)):
            path = join(path_dataset, folder_name)
            return path
    raise FileNotFoundError(
        f"None of the folders {folder_names} found in {path_dataset}"
    )


def get_rgbd_folders(path_dataset):
    path_color = add_if_exists(path_dataset, ["image/", "rgb/", "color/"])
    path_depth = join(path_dataset, "depth/")
    return path_color, path_depth


def get_rgbd_file_lists(path_dataset):
    path_color, path_depth = get_rgbd_folders(path_dataset)
    color_files = get_file_list(path_color, ".jpg") + get_file_list(path_color, ".png")
    depth_files = get_file_list(path_depth, ".png")
    return color_files, depth_files


def read_rgbd_image(color_file, depth_file, convert_rgb_to_intensity, config):
    color = o3d.io.read_image(color_file)
    depth = o3d.io.read_image(depth_file)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color,
        depth,
        depth_scale=config["depth_scale"],
        depth_trunc=config["depth_max"],
        convert_rgb_to_intensity=convert_rgb_to_intensity,
    )
    return rgbd_image


def scalable_integrate_rgb_frames(path_dataset, intrinsic, config):

    poses_path = path_dataset + "poses.txt"
    poses = np.loadtxt(poses_path)
    [color_files, depth_files] = get_rgbd_file_lists(path_dataset)
    n_files = len(color_files)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config["tsdf_cubic_size"] / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for frame_id in range(n_files):
        print(color_files[frame_id])
        print(depth_files[frame_id])
        rgbd = read_rgbd_image(
            color_files[frame_id], depth_files[frame_id], False, config
        )
        pose = poses[frame_id].reshape(4, 4)
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
        

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    if config["debug_mode"]:
        o3d.visualization.draw_geometries([mesh])

    mesh_name = join(path_dataset, "tsdf_mesh.ply")
    o3d.io.write_triangle_mesh(mesh_name, mesh, False, True)


def set_default_value(config, key, value):
    if key not in config:
        config[key] = value


def initialize_config(config):

    # set default parameters if not specified
    set_default_value(config, "depth_map_type", "redwood")
    set_default_value(config, "n_frames_per_fragment", 100)
    set_default_value(config, "n_keyframes_per_n_frame", 5)
    set_default_value(config, "depth_min", 0.3)
    set_default_value(config, "depth_max", 3.0)
    set_default_value(config, "depth_scale", 1000)
    set_default_value(config, "tsdf_cubic_size", 3.0)


def check_folder_structure(path_dataset):
    if isfile(path_dataset) and path_dataset.endswith(".bag"):
        return
    path_color, path_depth = get_rgbd_folders(path_dataset)
    assert exists(path_depth), "Path %s is not exist!" % path_depth
    assert exists(path_color), "Path %s is not exist!" % path_color


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruction system")
    parser.add_argument("--config", help="path to the config file", default=None)
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config) as json_file:
            config = json.load(json_file)
            initialize_config(config)
            check_folder_structure(config["path_dataset"])

    print("integrate the whole RGBD sequence using estimated camera pose.")
    if config["path_intrinsic"]:
        intrinsic = o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic"])
    else:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        )
    scalable_integrate_rgb_frames(config["path_dataset"], intrinsic, config)

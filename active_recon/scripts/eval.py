import os
import cv2
import numpy as np

from utils.config import load_parser
from utils.geometry import (
    marching_cubes,
    eval_surface_coverage_pcd,
    eval_entropy,
)
from utils.misc import load_model
from model.network import HashNeRF


def eval(args):

    bbox = [
        (-args.obj_size_guess, -args.obj_size_guess, 0 * args.obj_size_guess),
        (args.obj_size_guess, args.obj_size_guess, 2 * args.obj_size_guess),
    ]
    model_args = {"scale": args.obj_size_guess}
    # gt_meshfile = os.path.join(args.exp_path, "../../gt/bunny.ply")
    # gt_meshfile = os.path.join(args.exp_path, "../../gt/armadillo.ply")
    # gt_meshfile = os.path.join(args.exp_path, "../../gt/dragon.ply")
    # gt_meshfile = os.path.join(args.exp_path, "../../gt/gun.ply")
    # gt_meshfile = os.path.join(args.exp_path, "../../gt/car.ply")
    # gt_meshfile = os.path.join(args.exp_path, "../../gt/house.ply")
    gt_meshfile = os.path.join(args.exp_path, "../../gt/legotruck.ply")

    init_model = HashNeRF(**model_args).to(args.device)
    init_entropy, init_en_map_x, init_en_map_y, init_en_map_z = eval_entropy(
        init_model, args.mesh_reso, args.mesh_chunk, bbox, args.device
    )

    init_en_map_path_x = os.path.join(args.exp_path, "0_x.png")
    init_en_map_x = cv2.applyColorMap(init_en_map_x, cv2.COLORMAP_WINTER)
    cv2.imwrite(init_en_map_path_x, init_en_map_x)
    init_en_map_path_y = os.path.join(args.exp_path, "0_y.png")
    init_en_map_y = cv2.applyColorMap(init_en_map_y, cv2.COLORMAP_WINTER)
    cv2.imwrite(init_en_map_path_y, init_en_map_y)
    init_en_map_path_z = os.path.join(args.exp_path, "0_z.png")
    init_en_map_z = cv2.applyColorMap(init_en_map_z, cv2.COLORMAP_WINTER)
    cv2.imwrite(init_en_map_path_z, init_en_map_z)

    sc_list = []
    entropy_list = []

    for step in range(11):
        model_path = os.path.join(args.exp_path, str(step + 1) + ".pt")
        mesh_path = os.path.join(args.exp_path, str(step + 1) + ".ply")
        pcd_path = os.path.join(args.exp_path, str(step + 1) + "pcd.ply")

        model = load_model(model_path, args.hash, model_args).to(args.device)
        mesh = marching_cubes(
            model, args.mesh_reso, args.surf_thresh, args.mesh_chunk, bbox, args.device
        )
        mesh.remove_degenerate_faces()
        mesh.export(mesh_path)

        # sc = eval_surface_coverage(mesh_path, gt_meshfile, 0.005)
        sc_pcd = eval_surface_coverage_pcd(pcd_path, gt_meshfile, 0.005)
        entropy, entropy_map_x, entropy_map_y, entropy_map_z = eval_entropy(
            model, args.mesh_reso, args.mesh_chunk, bbox, args.device
        )

        en_map_path_x = os.path.join(args.exp_path, str(step + 1) + "_x.png")
        entropy_map_x = cv2.applyColorMap(entropy_map_x, cv2.COLORMAP_WINTER)
        cv2.imwrite(en_map_path_x, entropy_map_x)
        en_map_path_y = os.path.join(args.exp_path, str(step + 1) + "_y.png")
        entropy_map_y = cv2.applyColorMap(entropy_map_y, cv2.COLORMAP_WINTER)
        cv2.imwrite(en_map_path_y, entropy_map_y)
        en_map_path_z = os.path.join(args.exp_path, str(step + 1) + "_z.png")
        entropy_map_z = cv2.applyColorMap(entropy_map_z, cv2.COLORMAP_WINTER)
        cv2.imwrite(en_map_path_z, entropy_map_z)

        print("With the ", step + 1, "th view: ")
        # print("Surface Coverage: ", sc)
        print("Surface Coverage PCD: ", sc_pcd)
        print("Entropy: ", entropy)

        sc_list.append(sc_pcd)
        entropy_list.append(entropy)

    sc_list = np.array(sc_list)
    sc_file = os.path.join(args.exp_path, "sc.txt")
    np.savetxt(sc_file, sc_list)

    entropy_list = np.array(entropy_list)
    entropy_file = os.path.join(args.exp_path, "entropy.txt")
    np.savetxt(entropy_file, entropy_list)
    


if __name__ == "__main__":
    args = load_parser()

    # pose_path = os.path.join(args.exp_path, "kf_pose.pkl")
    # kf_pose = load_pose(pose_path)

    # model_path = os.path.join(args.exp_path, "model.pt")
    # model_args = {"scale": args.obj_size_guess}
    # model = load_model(model_path, args.hash, model_args).to(args.device)

    # bbox = [
    #     (-args.obj_size_guess, -args.obj_size_guess, 0.1 * args.obj_size_guess),
    #     (args.obj_size_guess, args.obj_size_guess, 1.9 * args.obj_size_guess),
    # ]

    # mesh_path = os.path.join(args.exp_path, "mesh.ply")
    # mesh = marching_cubes(
    #     model, args.mesh_reso, args.surf_thresh, args.mesh_chunk, bbox, args.device
    # )
    # mesh.remove_degenerate_faces()

    # pcd_path = os.path.join(args.exp_path, "pointcloud.ply")
    # pcd = o3d.io.read_point_cloud(pcd_path, format='ply')
    # # cull_mesh(mesh, pcd, 0.01)
    # mesh.export(mesh_path)

    eval(args)

    # visualizer = Visualizer(args)
    # visualizer.offline_vis(model)

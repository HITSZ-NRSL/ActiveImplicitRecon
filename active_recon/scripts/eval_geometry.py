import os
import numpy as np

from utils.config import load_parser
from utils.geometry import eval_mesh, cull_mesh

if __name__ == "__main__":
    args = load_parser()

    # gt_meshfile = os.path.join(args.exp_path, "../../gt/legotruck.ply")
    gt_meshfile = os.path.join(args.exp_path, "../../gt/car.ply")
    # gt_meshfile = os.path.join(args.exp_path, "../../gt/house.ply")

    pred_path = os.path.join(args.exp_path, "10.ply")
    tsdf_path = os.path.join(args.exp_path, "tsdf_mesh.ply")

    down_sample_vox = 0.001
    dist_thre = 0.01
    truncation_dist_acc = 0.05
    truncation_dist_com = 0.05

    geo_metrics = eval_mesh(
        pred_path,
        gt_meshfile,
        down_sample_res=down_sample_vox,
        threshold=dist_thre,
        truncation_acc=truncation_dist_acc,
        truncation_com=truncation_dist_com,
        gt_bbx_mask_on=True,
    )
    print(geo_metrics)

    geo_file = os.path.join(args.exp_path, "geometry.txt")
    geo_metrics = np.array([geo_metrics])
    np.savetxt(geo_file, geo_metrics, fmt="%s")

    # tsdf_geo_metrics = eval_mesh(
    #     tsdf_path,
    #     gt_meshfile,
    #     down_sample_res=down_sample_vox,
    #     threshold=dist_thre,
    #     truncation_acc=truncation_dist_acc,
    #     truncation_com=truncation_dist_com,
    #     gt_bbx_mask_on=True,
    # )
    # print(tsdf_geo_metrics)

    # tsdf_geo_file = os.path.join(args.exp_path, "tsdf_geometry.txt")
    # tsdf_geo_metrics = np.array([tsdf_geo_metrics])
    # np.savetxt(tsdf_geo_file, tsdf_geo_metrics, fmt="%s")

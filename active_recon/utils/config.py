import configargparse
import torch


def load_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument(
        "--config", is_config_file=True, default="config/nbv_select.txt"
    )
    parser.add_argument(
        "--exp_path", type=str, default="/home/star/Develop/ActiveRecon/exp/toycar3"
    )

    # Data params
    parser.add_argument("--device", type=int, default=0)

    parser.add_argument("--fx", type=float, default=564.3)
    parser.add_argument("--fy", type=float, default=564.3)
    parser.add_argument("--cx", type=float, default=480)
    parser.add_argument("--cy", type=float, default=270)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--width", type=int, default=960)

    parser.add_argument("--downsample_rate", type=float, default=1)
    parser.add_argument("--obj_size_guess", type=float, default=0.5)
    parser.add_argument("--max_depth_range", type=float, default=8)

    # Train params
    parser.add_argument("--global_pose_lr", type=float, default=2e-4)
    parser.add_argument("--network_lr", type=float, default=1e-3)
    parser.add_argument("--depth_loss_lambda", type=float, default=2)

    parser.add_argument("--rays_per_img_recon", type=int, default=1500)
    parser.add_argument("--recon_step", type=int, default=100)

    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--surf_points", type=int, default=32)
    parser.add_argument("--strat_points", type=int, default=8)

    parser.add_argument("--active_kf_size", type=int, default=5)
    parser.add_argument("--frame_buffer_len", type=int, default=30)
    parser.add_argument("--pixel_buffer_portion", type=float, default=0.1)

    # Eval params
    parser.add_argument("--mesh_chunk", type=int, default=8388608 / 2)
    parser.add_argument("--mesh_reso", type=int, default=256)
    parser.add_argument("--surf_thresh", type=float, default=0.5)

    parser.add_argument("--vis_iter_step", type=int, default=100)
    parser.add_argument("--vis_step", type=int, default=10)

    # Train options
    parser.add_argument("--hash", action="store_true", default=False)
    parser.add_argument("--pose_opt", action="store_true", default=False)

    # Eval options
    parser.add_argument("--verbose", action="store_true", default=False)

    parser.add_argument("--method", type=int, default=0)
    parser.add_argument("--disturb_pose", action="store_true", default=False)
    parser.add_argument("--save_vis", action="store_true", default=False)
    parser.add_argument("--mode", type=int, default=1)
    # parser.add_argument("--realworld", action="store_true", default=False)

    parser.add_argument(
        "--views_pose_path", type=str, default="config/dome_48_views.txt"
    )

    parser.add_argument("--linear_max", type=float, default=1.0)

    args = parser.parse_args()

    if args.device >= 0:
        args.device = torch.device("cuda", args.device)
    else:
        args.device = torch.device("cpu")

    args.height = int(args.height // args.downsample_rate)
    args.width = int(args.width // args.downsample_rate)

    args.fx = args.fx / args.downsample_rate
    args.fy = args.fy / args.downsample_rate
    args.cx = args.cx / args.downsample_rate
    args.cy = args.cy / args.downsample_rate

    return args

import random
import torch

from utils.geometry import get_intersect_point, sample_points_on_ray, sample_points_even_in_bbox

def render_occ_rgb(model,rays,z_vals,sample_points):
    rays_o, rays_d = rays[:, :3], rays[:, 3:6]
    rays_o = rays_o.view(-1, 1, 3)
    rays_d = rays_d.view(-1, 1, 3)
    xyz = rays_o + rays_d * z_vals.unsqueeze(-1)
    xyz = xyz.view(-1, 3)

    occs, rgbs = model(xyz)
    rgbs = rgbs.view(-1, sample_points, 3)
    occs = occs.view(-1, sample_points)
    return occs,rgbs

def render_color_depth_opacity_weight(occs,rgbs,z_vals):
    alphas = occs
    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)
    weights = alphas * torch.cumprod(alphas_shifted[:, :-1], -1)

    depths = torch.sum(weights * z_vals, 1)
    colors = torch.sum(weights.unsqueeze(-1) * rgbs, 1)
    opacities = torch.sum(weights, 1)
    return colors,depths,opacities,weights

def render_rays(
    model,
    rays,
    gt_depths,
    surf_points,
    strat_points,
    sigma,
    cube_half_size,
    lindisp=False,
    perturb=0.3,
    use_var=False,
):
    z_vals, sample_points, valid_mask ,void_mask= sample_points_on_ray(rays,gt_depths,surf_points,strat_points,
    sigma,
    cube_half_size,
    lindisp,
    perturb)
    
    occs, rgbs = render_occ_rgb(model, rays[valid_mask], z_vals, sample_points)
    colors, depths, opacities, weights = render_color_depth_opacity_weight(occs, rgbs, z_vals)
    
    if use_var:
        var = torch.sum(weights * torch.square(z_vals - depths.unsqueeze(1)), 1)
        return colors, depths, opacities, 1.0 / torch.sqrt(var)
    else:
        return colors, depths, opacities, valid_mask,void_mask

def render_void_rays(
    model, void_rays, sample_points, obj_size_guess, lindisp=False, perturb=0.3
):

    void_rays, z_vals = sample_points_even_in_bbox(void_rays, sample_points, obj_size_guess, lindisp, perturb)
    occ_void, _ = render_occ_rgb(model, void_rays, z_vals, sample_points)

    return occ_void

def render_color_only_rays(
    model,
    rays,
    sample_points,
    obj_size_guess,
    lindisp=False,
    perturb=0.3,
    use_var=False,
):
    rays_len = rays.shape[0]
    rays_o, rays_d = rays[:, :3], rays[:, 3:6]
    device = rays.device

    near, far, is_intersect = get_intersect_point(rays, obj_size_guess)
    if is_intersect.sum() != rays_len:
        raise Exception("Bounding box of obj_size_guess is too small!")

    near = near.unsqueeze(-1)
    far = far.unsqueeze(-1)
    t_vals = (
        torch.linspace(0, 1, steps=sample_points, device=device)
        .unsqueeze(0)
        .expand(rays_len, -1)
    )

    if not lindisp:
        z_vals = near * (1 - t_vals) + far * (t_vals)
    else:
        z_vals = 1 / (1 / near * (1 - t_vals) + 1 / far * (t_vals))

    if perturb > 0:
        z_vals_mid = 0.5 * (
            z_vals[:, :-1] + z_vals[:, 1:]
        )  # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        perturb_rand = perturb * torch.rand(z_vals.shape, device=device)
        z_vals = lower + (upper - lower) * perturb_rand

    rays_o = rays_o.view(-1, 1, 3)
    rays_d = rays_d.view(-1, 1, 3)
    xyz = rays_o + rays_d * z_vals.unsqueeze(-1)
    xyz = xyz.view(-1, 3)

    occs, rgbs = model(xyz)
    rgbs = rgbs.view(-1, sample_points, 3)
    occs = occs.view(-1, sample_points)

    alphas = occs
    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1)
    weights = alphas * torch.cumprod(alphas_shifted[:, :-1], -1)

    depths = torch.sum(weights * z_vals, 1)
    colors = torch.sum(weights.unsqueeze(-1) * rgbs, 1)
    opacities = torch.sum(weights, 1)
    
    if use_var:
        var = torch.sum(weights * torch.square(z_vals - depths.unsqueeze(1)), 1)
        return colors, depths, opacities, 1.0 / torch.sqrt(var)
    else:
        return colors, depths, opacities


def render_image(
    pose, model, bbox, gt_depth, intrinsic, surf_points, strat_points, sigma,cube_half_size
):
    # requires gt_mesh having no invalid value
    u_min, u_max = bbox[0]
    v_min, v_max = bbox[1]
    device = gt_depth.device

    gt_depths = gt_depth[v_min:v_max, u_min:u_max]
    gt_depths = gt_depths.view(-1)

    u = torch.arange(u_min, u_max, device = device)
    v = torch.arange(v_min, v_max, device = device)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    point_image = torch.stack([vv, uu], -1)
    height, width, _ = point_image.shape
    point_image = point_image.view(-1, 2)

    point_u = point_image[:, 1]
    point_v = point_image[:, 0]

    point_x = (point_u - intrinsic[0][2]) / intrinsic[0][0]
    point_y = (point_v - intrinsic[1][2]) / intrinsic[1][1]
    point_z = torch.ones_like(point_x)

    rays_d = torch.stack([point_x, point_y, point_z], -1)
    rays_o = torch.zeros_like(rays_d)
    rays = torch.cat([rays_o, rays_d], -1)

    rays = transform_ray(rays, pose)
    render_colors, render_depths, _ = render_rays(
        model, rays, gt_depths, surf_points, strat_points, sigma,cube_half_size
    )

    render_colors_image = render_colors.view(height, width, 3)
    render_depths_image = render_depths.view(height, width)

    return render_colors_image, render_depths_image


def render_synthetic_view(
    model, pose, bbox, n_sample, depth, intrinsic, surf_points, strat_points, sigma,cube_half_size
):
    # requires gt_mesh having no invalid value
    u_min, u_max = bbox[0]
    v_min, v_max = bbox[1]

    u = torch.arange(u_min, u_max)
    v = torch.arange(v_min, v_max)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    point_image = torch.stack([vv, uu], -1).to(pose.device)
    height, width, _ = point_image.shape
    point_image = point_image.view(-1, 2)

    point_u = point_image[:, 1]
    point_v = point_image[:, 0]

    intrin = intrinsic.to(pose.device)
    point_x = (point_u - intrin[0][2]) / intrin[0][0]
    point_y = (point_v - intrin[1][2]) / intrin[1][1]
    point_z = torch.ones_like(point_x)

    rays_d = torch.stack([point_x, point_y, point_z], -1)

    img_size = (340 - 160) * (460 - 220)
    if img_size < n_sample:
        rays_ind = torch.randint(0, img_size, (n_sample,))
    else:
        rays_ind = random.sample(range(img_size), n_sample)

    rays_d = rays_d[rays_ind]
    rays_o = torch.zeros_like(rays_d)
    rays = torch.cat([rays_o, rays_d], -1)

    rays = transform_ray(rays, pose)

    depths = torch.ones(rays.shape[0], device=rays.device) * depth
    render_colors, render_depths, _, invsqrt_var = render_rays(
        model, rays, depths, surf_points, strat_points, sigma,cube_half_size, use_var=True
    )

    return render_colors, render_depths, invsqrt_var

def render_normal(rays, gt_depths, model, epsilon):
    sample_rays = rays.shape[0]
    rays_o, rays_d = rays[:, :3], rays[:, 3:6]
    device = rays.device
    
    xyz = rays_o + rays_d * gt_depths.unsqueeze(-1)
    xyz = xyz.view(-1, 3)
    
    xyz_neighbor = xyz + torch.randn_like(xyz, device = device) * epsilon
    
    xyz_full = torch.concat([xyz, xyz_neighbor], 0)
    
    gradient_full = model.gradient(xyz_full)
    gradient_full = gradient_full.squeeze(1) / torch.norm(gradient_full, dim = -1)
    gradient = gradient_full[:sample_rays]
    gradient_neighbor = gradient_full[sample_rays:]
    
    return gradient, gradient_neighbor

def transform_ray(rays, pose):
    rays_o = rays[:, :3] + pose[:3, 3]
    rays_d = pose[:3, :3] @ rays[:, 3:6].transpose(1, 0)
    rays_d = rays_d.transpose(1, 0)

    rays = torch.cat([rays_o, rays_d], -1)

    return rays

def gen_random_rays(width, height, n_sample, intrinsic, device = "cpu"):
    point_u = torch.randint(0, width, (n_sample,), device = device)
    point_v = torch.randint(0, height, (n_sample,), device = device)

    point_x = (point_u - intrinsic[0][2]) / intrinsic[0][0]
    point_y = (point_v - intrinsic[1][2]) / intrinsic[1][1]
    point_z = torch.ones_like(point_x)

    rays_d = torch.stack([point_x, point_y, point_z], -1)

    rays_o = torch.zeros_like(rays_d)
    rays = torch.cat([rays_o, rays_d], -1)

    return rays
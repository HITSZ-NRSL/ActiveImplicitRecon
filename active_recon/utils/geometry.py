import random
import numpy as np
import cv2
import torch
import mcubes
import trimesh
import open3d as o3d
from scipy.spatial import KDTree


def sample_points_even_in_bbox(
    rays, sample_points, obj_size_guess, lindisp=False, perturb=0.3
):
    device = rays.device

    near, far, is_intersect = get_intersect_point(rays, obj_size_guess)

    near = near[is_intersect].unsqueeze(-1)
    far = far[is_intersect].unsqueeze(-1)

    rays = rays[is_intersect]
    rays_len = rays.shape[0]

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

    return rays, z_vals


def sample_points_on_ray(
    rays,
    gt_depths,
    surf_points,
    strat_points,
    sigma,
    cube_half_size,
    lindisp=False,
    perturb=0.3,
):
    rays_len = rays.shape[0]
    device = rays.device

    # clamp_min：设置一个下限min，tensor中有元素小于这个值, 就把对应的值赋为min
    # near = torch.clamp_min(gt_depths - 3 * sigma, 0.0).unsqueeze(-1)
    # far = (gt_depths + 3 * sigma).unsqueeze(-1)

    near, far, is_intersect = get_intersect_point(rays, cube_half_size)
    valid_depth_mask = gt_depths < far
    valid_mask = valid_depth_mask & is_intersect
    void_mask = ~valid_depth_mask & is_intersect

    near = near[valid_mask].unsqueeze(-1)
    far = far[valid_mask].unsqueeze(-1)
    gt_depths = gt_depths[valid_mask]

    rays_len = gt_depths.shape[0]

    z_vals_surf = torch.randn(
        (rays_len, surf_points), device=device
    ) * sigma + gt_depths.view(-1, 1)
    z_vals_surf = torch.clamp(z_vals_surf, near, far)
    # far = gt_depths.unsqueeze(1)
    # near = gt_depths.unsqueeze(1)

    if strat_points > 0:
        t_vals_strat = (
            torch.linspace(0, 1, steps=strat_points, device=device)
            .unsqueeze(0)
            .expand(rays_len, -1)
        )
        if not lindisp:
            z_vals_strat = near * (1 - t_vals_strat) + far * (t_vals_strat)
        else:
            z_vals_strat = 1 / (
                1 / near * (1 - t_vals_strat) + 1 / far * (t_vals_strat)
            )

        z_vals = torch.cat([z_vals_surf, z_vals_strat], -1)
        sample_points = surf_points + strat_points

        if perturb > 0:
            z_vals_mid = 0.5 * (
                z_vals[:, :-1] + z_vals[:, 1:]
            )  # (N_rays, N_samples-1) interval mid points
            # get intervals between samples
            upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
            lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

            perturb_rand = perturb * torch.rand(z_vals.shape, device=device)
            z_vals = lower + (upper - lower) * perturb_rand

    else:
        z_vals = z_vals_surf
        sample_points = surf_points

    z_vals = torch.sort(z_vals, -1).values
    return z_vals, sample_points, valid_mask, void_mask


def marching_cubes(model, reso, surf_thresh, chunk, bbox, device):
    x_min, y_min, z_min = bbox[0]
    x_max, y_max, z_max = bbox[1]

    scale = reso / (x_max - x_min)
    offset = np.array([x_min, y_min, z_min])

    step_x = reso
    step_y = round(scale * (y_max - y_min))
    step_z = round(scale * (z_max - z_min))

    y_max = y_min + step_y / scale
    z_max = z_min + step_z / scale

    x = torch.linspace(x_min, x_max, step_x, device=device)
    y = torch.linspace(y_min, y_max, step_y, device=device)
    z = torch.linspace(z_min, z_max, step_z, device=device)

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
    grid = torch.stack([grid_x, grid_y, grid_z], -1)
    points = grid.view(-1, 3)

    # Extract vertices and faces using marching cubes
    model_out = []
    with torch.no_grad():
        for chunk_num in range(0, points.shape[0], chunk):
            points_chunk = points[chunk_num : chunk_num + chunk]

            model_out_chunk = model(points_chunk)[0]

            model_out_chunk = model_out_chunk.detach().cpu()
            model_out.append(model_out_chunk)

        model_out = torch.cat(model_out, 0)
        model_out = model_out.view(step_x, step_y, step_z)

    vertices, triangles = mcubes.marching_cubes(
        model_out.numpy().astype(np.float32), surf_thresh
    )
    vertices = vertices / scale + offset

    # Extract colors at vertices
    points = torch.from_numpy(vertices).to(device)
    model_out = []
    with torch.no_grad():
        for chunk_num in range(0, points.shape[0], chunk):
            points_chunk = points[chunk_num : chunk_num + chunk]

            model_out_chunk = model(points_chunk)[1]

            model_out_chunk = model_out_chunk.detach().cpu()
            model_out.append(model_out_chunk)

        model_out = torch.cat(model_out, 0)
        vertex_colors = model_out.view(-1, 3).numpy().astype(np.float32)
        vertex_colors = (vertex_colors * 255).astype(np.uint8)

    mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=vertex_colors)

    # # Extract uncertainty at vertices (represent using colormap)
    # points = torch.from_numpy(vertices).to(device)
    # model_out = []
    # with torch.no_grad():
    #     for chunk_num in range(0, points.shape[0], chunk):
    #         points_chunk = points[chunk_num : chunk_num + chunk]

    #         model_out_chunk = ig  # TODO add uncertainty head

    #         model_out_chunk = model_out_chunk.detach().cpu()
    #         model_out.append(model_out_chunk)

    #     model_out = torch.cat(model_out, 0)
    #     vertex_uncertainty = model_out.view(-1, 3).numpy().astype(np.float32)
    #     vertex_uncertainty = (vertex_uncertainty * 255).astype(np.uint8)
    #     cv2.applyColorMap(vertex_uncertainty, vertex_uncertainty, cv2.COLORMAP_WINTER)

    # mesh_uncertainty = trimesh.Trimesh(
    #     vertices, triangles, vertex_colors=vertex_uncertainty
    # )

    return mesh


def incam(points, pose, K, H, W):

    points = torch.cat([points, torch.ones(1, points.shape[1])], dim=0)
    cam_points = (pose @ points)[:3]
    image_points = K @ cam_points
    image_points = image_points[:2] / image_points[2]

    inmask = (
        (image_points[0, :] < W)
        & (image_points[0, :] > 0.0)
        & (image_points[1, :] < H)
        & (image_points[1, :] > 0.0)
    )

    return inmask


# 获得立方体与线段 line_segment 的两个交点
# Need rays_d to obey z = 1 (before transform)
def get_intersect_point(rays, cube_half_size):
    rays_o, rays_d = rays[:, :3], rays[:, 3:6]
    max_xyz = torch.tensor(
        [cube_half_size, cube_half_size, 2 * cube_half_size], device=rays_d.device
    )
    min_xyz = torch.tensor(
        [-cube_half_size, -cube_half_size, 0 * cube_half_size], device=rays_d.device
    )

    # Force the parallel rays to intersect with plane, and they will be removed by sanity check
    mask_parallels = rays_d == 0
    rays_d[mask_parallels] = rays_d[mask_parallels] + 1e-6

    z_nears = (min_xyz - rays_o) / rays_d
    z_fars = (max_xyz - rays_o) / rays_d

    mask_exchange = z_nears > z_fars
    z_nears[mask_exchange], z_fars[mask_exchange] = (
        z_fars[mask_exchange],
        z_nears[mask_exchange],
    )

    z_near = torch.max(z_nears, dim=1).values
    z_far = torch.min(z_fars, dim=1).values

    # sanity check
    mask_hits = z_near < z_far
    tmp_mask = z_near < 0
    z_near[tmp_mask] = 0

    return z_near, z_far, mask_hits


def segment_plane_rsc(pts, thresh=0.05, maxIter=10):
    """
    分割平面
    :param pts: 点云
    :param thresh: 阈值
    :param maxIter: 最大迭代次数
    :return:
    """
    n_points = pts.shape[0]
    best_eq = []
    best_inliers = []

    for it in range(maxIter):

        # Samples 3 random points
        id_samples = random.sample(range(0, n_points), 3)
        pt_samples = pts[id_samples]

        # We have to find the plane equation described by those 3 points
        # We find first 2 vectors that are part of this plane
        # A = pt2 - pt1
        # B = pt3 - pt1

        vecA = pt_samples[1, :] - pt_samples[0, :]
        vecB = pt_samples[2, :] - pt_samples[0, :]

        # Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
        vecC = torch.cross(vecA, vecB)

        # The plane equation will be vecC[0]*x + vecC[1]*y + vecC[0]*z = -k
        # We have to use a point to find k
        vecC = vecC / vecC.norm()
        k = -(torch.mul(vecC, pt_samples[1, :])).sum()
        plane_eq = [vecC[0], vecC[1], vecC[2], k]

        # Distance from a point to a plane
        # https://mathworld.wolfram.com/Point-PlaneDistance.html
        pt_id_inliers = []  # list of inliers ids
        dist_pt = (
            plane_eq[0] * pts[:, 0]
            + plane_eq[1] * pts[:, 1]
            + plane_eq[2] * pts[:, 2]
            + plane_eq[3]
        ) / torch.sqrt(plane_eq[0] ** 2 + plane_eq[1] ** 2 + plane_eq[2] ** 2)

        # Select indexes where distance is biggers than the threshold
        pt_id_inliers = torch.where(torch.abs(dist_pt) <= thresh)[0]
        if len(pt_id_inliers) > len(best_inliers):
            best_eq = plane_eq
            best_inliers = pt_id_inliers

    return best_eq, best_inliers


def segment_plane_rsc_acc(pts, thresh=0.05, maxIter=10):
    """
    分割平面
    :param pts: 点云
    :param thresh: 阈值
    :param maxIter: 最大迭代次数
    :return:
    """
    n_points = pts.shape[0]

    # Samples 3 random points
    n_sample_points = 3 * maxIter
    id_samples = random.sample(range(0, n_points), n_sample_points)
    pt_samples = pts[id_samples]

    # We have to find the plane equation described by those 3 points
    # We find first 2 vectors that are part of this plane
    # A = pt2 - pt1
    # B = pt3 - pt2
    vecAB = pt_samples[1:n_sample_points, :] - pt_samples[: n_sample_points - 1, :]

    # Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
    vecC = torch.cross(
        vecAB[: vecAB.shape[0] - 1, :], vecAB[1 : vecAB.shape[0], :], dim=1
    )

    # The plane equation will be vecC[0]*x + vecC[1]*y + vecC[2]*z = -k
    # We have to use a point to find k
    vecC = vecC / vecC.norm(dim=1).reshape(-1, 1)
    k = -(torch.mul(vecC, pt_samples[: vecC.shape[0], :])).sum(dim=1)
    plane_eq = torch.cat([vecC, k.reshape(-1, 1)], dim=1)

    # Distance from a point to a plane
    # https://mathworld.wolfram.com/Point-PlaneDistance.html
    pts = pts.transpose(1, 0)
    ones = torch.ones(1, pts.shape[1], device=pts.device)
    pts = torch.cat([pts, ones], dim=0)

    dist_pts = (plane_eq @ pts) / torch.sqrt(
        plane_eq[:, 0] ** 2 + plane_eq[:, 1] ** 2 + plane_eq[:, 2] ** 2
    ).reshape(-1, 1)

    # Select indexes where distance is biggers than the threshold
    mask_inliers = torch.abs(dist_pts) <= thresh
    n_inliers = torch.sum(mask_inliers, dim=1)
    _, model_id = torch.max(n_inliers, 0)

    best_eq = plane_eq[model_id]
    best_inliers = torch.nonzero(mask_inliers[model_id, :]).squeeze()
    best_ouliers = torch.nonzero(~mask_inliers[model_id, :]).squeeze()

    return best_eq, best_inliers, best_ouliers


def secant(
    model,
    f_low,
    f_high,
    d_low,
    d_high,
    n_secant_steps,
    ray0_masked,
    ray_direction_masked,
    tau,
    it=0,
):
    """Runs the secant method for interval [d_low, d_high].

    Args:
        d_low (tensor): start values for the interval
        d_high (tensor): end values for the interval
        n_secant_steps (int): number of steps
        ray0_masked (tensor): masked ray start points
        ray_direction_masked (tensor): masked ray direction vectors
        model (nn.Module): model model to evaluate point occupancies
        c (tensor): latent conditioned code c
        tau (float): threshold value in logits
    """
    d_pred = -f_low * (d_high - d_low) / (f_high - f_low) + d_low
    for i in range(n_secant_steps):
        p_mid = ray0_masked + d_pred.unsqueeze(-1) * ray_direction_masked

        with torch.no_grad():
            f_mid = model(p_mid)[0] - tau
        ind_low = f_mid < 0
        ind_low = ind_low
        if ind_low.sum() > 0:
            d_low[ind_low] = d_pred[ind_low]
            f_low[ind_low] = f_mid[ind_low]
        if (ind_low == 0).sum() > 0:
            d_high[ind_low == 0] = d_pred[ind_low == 0]
            f_high[ind_low == 0] = f_mid[ind_low == 0]

        d_pred = -f_low * (d_high - d_low) / (f_high - f_low) + d_low

    return d_pred


def ray_marching(
    rays,
    model,
    tau=0.5,
    n_steps=128,
    n_secant_steps=8,
    depth_range=[0.0, 2.4],
    chunk=3500000,
):
    """Performs ray marching to detect surface points.

    The function returns the surface points as well as d_i of the formula
        ray(d_i) = ray_o + d_i * ray_d
    which hit the surface points. In addition, masks are returned for
    illegal values.

    Args:
        rays (tensor): ray start points and direction N x (3 + 3)
        model (nn.Module): model model to evaluate point occupancies
        tau (float): threshold value
        n_steps (tuple): interval from which the number of evaluation
            steps if sampled
        n_secant_steps (int): number of secant refinement steps
        depth_range (tuple): range of possible depth values (not relevant when
            using cube intersection)
        max_points (int): max number of points loaded to GPU memory
    """
    # Shotscuts
    rays_len = rays.shape[0]
    device = rays.device
    rays_o = rays[:, :3]
    rays_d = rays[:, 3:6]

    d_proposal = torch.linspace(0, 1, steps=n_steps, device=device).view(1, n_steps)
    d_proposal = depth_range[0] * (1.0 - d_proposal) + depth_range[1] * d_proposal
    d_proposal = d_proposal.expand(rays_len, -1)

    p_proposal = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * d_proposal.unsqueeze(-1)
    p_proposal = p_proposal.view(-1, 3)  # [rays_len x n_steps, 3]

    # Evaluate all proposal points in parallel
    with torch.no_grad():
        val = []
        for chunk_num in range(0, p_proposal.shape[0], chunk):
            p_proposal_chunk = p_proposal[chunk_num : chunk_num + chunk]

            val_chunk = model(p_proposal_chunk)[0] - tau
            val.append(val_chunk)

        val = torch.cat(val, 0).view(rays_len, -1)  # [rays_len, n_steps]

    # Create mask for valid points where the first point is not occupied
    mask_0_not_occupied = val[:, 0] < 0

    # Calculate if sign change occurred and concat 1 (no sign change) in last dimension
    signs = torch.cat(
        [
            torch.sign(val[:, :-1] * val[:, 1:]),
            torch.ones((rays_len, 1), device=device),
        ],
        -1,
    )
    costs = signs * torch.arange(0, 1, n_steps, device=device, dtype=float)

    # Get first sign change and mask for values where
    # a.) a sign changed occurred and
    # b.) no a neg to pos sign change occurred (meaning from inside surface to outside)
    values, indices = torch.min(costs, -1)
    mask_sign_change = values < 0

    val_at_ind = torch.gather(val, 1, indices.unsqueeze(-1))
    d_proposal_at_ind = torch.gather(d_proposal, 1, indices.unsqueeze(-1))
    mask_neg_to_pos = (val_at_ind < 0).squeeze(-1)

    # Define mask where a valid depth value is found
    mask = mask_sign_change & mask_neg_to_pos & mask_0_not_occupied

    # Get depth values and function values for the interval
    # to which we want to apply the Secant method
    d_low = d_proposal_at_ind[mask]
    f_low = val_at_ind[mask]

    indices = torch.clamp(indices + 1, max=n_steps - 1)
    val_at_ind = torch.gather(val, 0, indices.unsqueeze(-1))
    d_proposal_at_ind = torch.gather(d_proposal, 0, indices.unsqueeze(-1))

    d_high = d_proposal_at_ind[mask]
    f_high = val_at_ind[mask]

    rays_o_masked = rays_o[mask]
    rays_d_masked = rays_d[mask]

    # Apply surface depth refinement step (e.g. Secant method)
    d_pred = secant(
        model,
        f_low,
        f_high,
        d_low,
        d_high,
        n_secant_steps,
        rays_o_masked,
        rays_d_masked,
        tau,
    )

    # for sanity
    d_pred_out = torch.ones(rays_len, device=device)
    d_pred_out[mask] = d_pred
    d_pred_out[mask == 0] = np.inf
    d_pred_out[mask_0_not_occupied == 0] = 0

    return d_pred_out


def register_pcd(pcd, rays, rays_depth, scale):
    # random down sample
    rays_o = rays[:, :3]
    rays_d = rays[:, 3:6]
    device = rays_o.device

    points = rays_o + rays_depth.unsqueeze(-1) * rays_d
    min_mask = points < torch.tensor([[scale, scale, 2 * scale]], device=device)
    max_mask = points > torch.tensor([[-scale, -scale, 0 * scale]], device=device)
    mask_mask = min_mask & max_mask
    mask_mask_mask = mask_mask[:, 0] & mask_mask[:, 1] & mask_mask[:, 2]
    points = points[mask_mask_mask]
    points = points.cpu().numpy()

    pcd_new = o3d.geometry.PointCloud()
    pcd_new.points = o3d.utility.Vector3dVector(points)

    pcd_new = pcd + pcd_new
    pcd_new = pcd_new.voxel_down_sample(voxel_size=0.0005)

    return pcd_new


def cull_mesh(mesh, pcd, ball_size):
    mesh_pcd = mesh.vertices
    pcd_np = np.asarray(pcd.points)

    mesh_points_kd_tree = KDTree(mesh_pcd)
    idx = mesh_points_kd_tree.query_ball_point(pcd_np, ball_size)

    idx = np.concatenate(idx).astype(np.int32)

    vertice_mask = np.zeros(mesh_pcd.shape[0]).astype(bool)
    vertice_mask[idx] = True
    face_mask = vertice_mask[mesh.faces].all(axis=1)

    mesh.update_vertices(vertice_mask)
    mesh.update_faces(face_mask)

    return mesh


def get_align_transformation(o3d_rec_mesh, o3d_gt_mesh):
    """
    Get the transformation matrix to align the reconstructed mesh to the ground truth mesh.
    """
    o3d_rec_pc = o3d.geometry.PointCloud(points=o3d_rec_mesh.vertices)
    o3d_gt_pc = o3d.geometry.PointCloud(points=o3d_gt_mesh.vertices)
    # trans_init = np.eye(4)
    trans_init = np.linalg.inv(
        np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    )
    threshold = 0.1

    reg_p2p = o3d.pipelines.registration.registration_icp(
        o3d_rec_pc,
        o3d_gt_pc,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )

    transformation = reg_p2p.transformation
    return transformation


def nn_correspondance(verts1, verts2, truncation_dist, ignore_outlier=True):
    """for each vertex in verts2 find the nearest vertex in verts1
    Args:
        nx3 np.array's
        scalar truncation_dist: points whose nearest neighbor is farther than the distance would not be taken into account
    Returns:
        ([indices], [distances])
    """

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts1)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    truncation_dist_square = truncation_dist**2

    for vert in verts2:
        _, inds, dist_square = kdtree.search_knn_vector_3d(vert, 1)

        if dist_square[0] < truncation_dist_square:
            indices.append(inds[0])
            distances.append(np.sqrt(dist_square[0]))
        else:
            if not ignore_outlier:
                indices.append(inds[0])
                distances.append(truncation_dist)

    return indices, distances


def eval_surface_coverage(rec_meshfile, gt_meshfile, ball_size):
    o3d_rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    o3d_gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)

    # transformation = get_align_transformation(o3d_rec_mesh, o3d_gt_mesh)
    # o3d_rec_mesh = o3d_rec_mesh.transform(transformation)

    o3d_rec_pcd = np.array(o3d_rec_mesh.vertices)
    o3d_gt_pcd = np.array(o3d_gt_mesh.vertices)

    gt_kd_tree = KDTree(o3d_gt_pcd)
    idx = gt_kd_tree.query_ball_point(o3d_rec_pcd, ball_size)
    idx = np.concatenate(idx).astype(np.int32)

    pcd_list = np.zeros(o3d_gt_pcd.shape[0])
    pcd_list[idx] = 1

    surface_coverage = pcd_list.sum() / o3d_gt_pcd.shape[0]

    return surface_coverage


def eval_surface_coverage_pcd(
    rec_pcdfile, gt_meshfile, ball_size, pos_bias=np.array([0, 0, 0])
):
    o3d_rec_pcd = o3d.io.read_point_cloud(rec_pcdfile)
    o3d_gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)

    # transformation = get_align_transformation(o3d_rec_mesh, o3d_gt_mesh)
    # o3d_rec_mesh = o3d_rec_mesh.transform(transformation)

    o3d_rec_pcd = np.array(o3d_rec_pcd.points)
    o3d_gt_pcd = np.array(o3d_gt_mesh.vertices) + pos_bias

    gt_kd_tree = KDTree(o3d_gt_pcd)
    idx = gt_kd_tree.query_ball_point(o3d_rec_pcd, ball_size)
    idx = np.concatenate(idx).astype(np.int32)

    pcd_list = np.zeros(o3d_gt_pcd.shape[0])
    pcd_list[idx] = 1

    surface_coverage = pcd_list.sum() / o3d_gt_pcd.shape[0]

    return surface_coverage


def eval_entropy(model, reso, chunk, bbox, device):
    x_min, y_min, z_min = bbox[0]
    x_max, y_max, z_max = bbox[1]

    scale = reso / (x_max - x_min)
    offset = np.array([x_min, y_min, z_min])

    step_x = reso
    step_y = round(scale * (y_max - y_min))
    step_z = round(scale * (z_max - z_min))

    y_max = y_min + step_y / scale
    z_max = z_min + step_z / scale

    x = torch.linspace(x_min, x_max, step_x, device=device)
    y = torch.linspace(y_min, y_max, step_y, device=device)
    z = torch.linspace(z_min, z_max, step_z, device=device)

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
    grid = torch.stack([grid_x, grid_y, grid_z], -1)
    points = grid.view(-1, 3)

    # Extract vertices and faces using marching cubes
    model_out = []
    with torch.no_grad():
        for chunk_num in range(0, points.shape[0], chunk):
            points_chunk = points[chunk_num : chunk_num + chunk]

            model_out_chunk = model(points_chunk)[0]

            model_out_chunk = model_out_chunk.detach().cpu()
            model_out.append(model_out_chunk)

        model_out = torch.cat(model_out, 0)
        model_out = model_out.view(step_x, step_y, step_z)

    entropy = -model_out * torch.log2(model_out + 1e-6) - (1 - model_out) * torch.log2(
        1 - model_out + 1e-6
    )

    entropy_map_x = entropy.max(0).values.numpy().astype(np.uint8)
    entropy_map_y = entropy.max(1).values.numpy().astype(np.uint8)
    entropy_map_z = entropy.max(2).values.numpy().astype(np.uint8)

    entropy_map_x = entropy.sum(0).numpy().astype(np.uint8)
    entropy_map_y = entropy.sum(1).numpy().astype(np.uint8)
    entropy_map_z = entropy.sum(2).numpy().astype(np.uint8)

    entropy = entropy.mean()

    return entropy.item(), entropy_map_x, entropy_map_y, entropy_map_z


def eval_mesh(
    file_pred,
    file_trgt,
    down_sample_res=0.02,
    threshold=0.05,
    truncation_acc=0.50,
    truncation_com=0.50,
    gt_bbx_mask_on=True,
    mesh_sample_point=10000000,
    possion_sample_init_factor=5,
):
    """Compute Mesh metrics between prediction and target.
    Opens the Meshs and runs the metrics
    Args:
        file_pred: file path of prediction (should be mesh)
        file_trgt: file path of target (shoud be point cloud)
        down_sample_res: use voxel_downsample to uniformly sample mesh points
        threshold: distance threshold used to compute precision/recall
        truncation_acc: points whose nearest neighbor is farther than the distance would not be taken into account (take pred as reference)
        truncation_com: points whose nearest neighbor is farther than the distance would not be taken into account (take trgt as reference)
        gt_bbx_mask_on: use the bounding box of the trgt as a mask of the pred mesh
        mesh_sample_point: number of the sampling points from the mesh
        possion_sample_init_factor: used for possion uniform sampling, check open3d for more details (deprecated)
    Returns:

    Returns:
        Dict of mesh metrics (chamfer distance, precision, recall, f1 score, etc.)
    """

    mesh_pred = o3d.io.read_triangle_mesh(file_pred)
    pcd_trgt = o3d.io.read_point_cloud(file_trgt)

    # (optional) filter the prediction outside the gt bounding box (since gt sometimes is not complete enough)
    if gt_bbx_mask_on:
        trgt_bbx = pcd_trgt.get_axis_aligned_bounding_box()
        min_bound = trgt_bbx.get_min_bound() - 2 * down_sample_res
        min_bound[2] += 2 * down_sample_res
        max_bound = trgt_bbx.get_max_bound() + 2 * down_sample_res
        # max_bound[2]+=down_sample_res
        trgt_bbx = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        mesh_pred = mesh_pred.crop(trgt_bbx)

    # o3d.visualization.draw_geometries([mesh_pred])

    # pcd_sample_pred = mesh_pred.sample_points_poisson_disk(number_of_points=mesh_sample_point, init_factor=possion_sample_init_factor)
    # mesh uniform sampling
    pcd_sample_pred = mesh_pred.sample_points_uniformly(
        number_of_points=mesh_sample_point
    )

    if down_sample_res > 0:
        pred_pt_count_before = len(pcd_sample_pred.points)
        pcd_pred = pcd_sample_pred.voxel_down_sample(down_sample_res)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample_res)
        pred_pt_count_after = len(pcd_pred.points)
        print(
            "Predicted mesh unifrom sample: ",
            pred_pt_count_before,
            " --> ",
            pred_pt_count_after,
            " (",
            down_sample_res,
            "m)",
        )

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    _, dist_p = nn_correspondance(
        verts_trgt, verts_pred, truncation_acc, True
    )  # find nn in ground truth samples for each predict sample -> precision related
    _, dist_r = nn_correspondance(
        verts_pred, verts_trgt, truncation_com, False
    )  # find nn in predict samples for each ground truth sample -> recall related

    dist_p = np.array(dist_p)
    dist_r = np.array(dist_r)

    dist_p_s = np.square(dist_p)
    dist_r_s = np.square(dist_r)

    dist_p_mean = np.mean(dist_p)
    dist_r_mean = np.mean(dist_r)

    dist_p_s_mean = np.mean(dist_p_s)
    dist_r_s_mean = np.mean(dist_r_s)

    chamfer_l1 = 0.5 * (dist_p_mean + dist_r_mean)
    chamfer_l2 = np.sqrt(0.5 * (dist_p_s_mean + dist_r_s_mean))

    precision = np.mean((dist_p < threshold).astype("float")) * 100.0  # %
    recall = np.mean((dist_r < threshold).astype("float")) * 100.0  # %
    fscore = 2 * precision * recall / (precision + recall)  # %

    metrics = {
        "MAE_accuracy (m)": dist_p_mean,
        "MAE_completeness (m)": dist_r_mean,
        # "Chamfer_L1 (m)": chamfer_l1,
        # "Chamfer_L2 (m)": chamfer_l2,
        # "Precision [Accuracy] (%)": precision,
        "Recall [Completeness] (%)": recall,
        # "F-score (%)": fscore,
        # "Spacing (m)": down_sample_res,  # evlaution setup
        # "Inlier_threshold (m)": threshold,  # evlaution setup
        # "Outlier_truncation_acc (m)": truncation_acc,  # evlaution setup
        # "Outlier_truncation_com (m)": truncation_com,  # evlaution setup
    }
    return metrics

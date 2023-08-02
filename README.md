# ActiveImplicitRecon

# Active Implicit Object Reconstruction using Uncertainty-guided Next-Best-View Optimization

## 1. Introduction
In this work, we propose a seamless integration of the emerging implicit representation with the active reconstruction task.
We build an implicit occupancy field as our geometry proxy.
While training, the prior object bounding box is utilized as auxiliary information to generate clean and detailed reconstructions.
To evaluate view uncertainty, we employ a sampling-based approach that directly extracts entropy from the reconstructed occupancy probability field as our measure of view information gain.
This eliminates the need for additional uncertainty maps or learning.
Unlike previous methods that compare view uncertainty within a finite set of candidates, we aim to find the next-best-view (NBV) on a continuous manifold.
Leveraging the differentiability of the implicit representation, the NBV can be optimized directly by maximizing the view uncertainty using gradient descent.
It significantly enhances the method's adaptability to different scenarios.

Authors: [Dongyu Yan](https://github.com/StarRealMan)\*, [Jianheng Liu](https://github.com/jianhengLiu)\*, [Fengyu Quan](https://github.com/jianhengLiu), [Haoyao Chen](https://github.com/HitszChen), and Mengmeng Fu.
\* Equal contribution.


If you use ActiveImplicitRecon for your academic research, please cite the following paper [[pdf](https://arxiv.org/abs/2303.16739)]. 
```
@article{yan2023active,
  title={Active Implicit Object Reconstruction using Uncertainty-guided Next-Best-View Optimziation},
  author={Yan, Dongyu and Liu, Jianheng and Quan, Fengyu and Chen, Haoyao and Fu, Mengmeng},
  journal={arXiv preprint arXiv:2303.16739},
  year={2023}
}
```

# 2. Usage

## Prerequisites

1. clone repo
    ```bash
    mkdir -p AiR_ws/src
    cd AiR_ws/src
    git clone https://github.com/HITSZ-NRSL/ActiveImplicitRecon.git
    cd ..
    catkin_make
    source devel/setup.bash
    # source devel/setup.zsh
    ```

2. Create envireonment using 
    ```
    $ conda env create -f active_recon/environment.yml
    $ conda activate active recon
    ```

3. Install ROS and related packages
    ```
    $ pip install pyyaml
    $ pip install rospkg
    ```

4. Install tiny-cuda-nn and apex following the instructions in [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) and [apex](https://github.com/NVIDIA/apex)

## Simulation

1. start simulation
```bash
export GAZEBO_MODEL_PATH=(path_to_repo)/ActiveImplicitRecon/gazebo_simulation/model:$GAZEBO_MODEL_PATH

roslaunch active_recon_gazebo simulation_bunny.launch
```
2. visualization

```bash
roslaunch active_recon_gazebo rviz.launch
```

3. (optional) reset simulation
```bash
sh src/active_recon_gazebo/scripts/reset_model.bash
```


## ActiveImplicitRecon
```bash
python main.py --hash --config config/gazebo.txt --exp_path active_recon/exp/test 
```
- different methods:
  
  0. (default) ours sample_seed+end2end
  1. end2end
  2. candidate views
  3. random views sphere
  4. random views shell
  5. circular
  6. new sample_seed+end2end

- mode:
  
  0. (default) offline
  1. simulation
  2. realworld


- pose opt: --pose_opt
- save vis output: --save_vis
- add disturb to poses: --disturb_pose

for more options, please refer to config/gazebo.txt

# Evaluation

1. surface coverage and entropy evaluation
    ```bash
    # !!! make sure to change the gt mesh path in eval.py and eval_geometry.py to the correct path
    # mesh generation and surface coverage/entropy evaluation: sc.txt and entropy.txt
    python scripts/eval.py --hash --exp_path active_recon/exp --config config/gazebo.txt
    ```

2. geometry metrics evaluation
    ```bash
    # geometry evaluation: results saved in geometry.txt and tsdf_geometry.txt
    python scripts/eval_geometry.py --hash --exp_path active_recon/exp --config config/gazebo.txt
    ```

3. (optional) tsdf-based mapping
    ```bash
    # TSDF reconstruction: tsdf_mesh.ply, after active_recon, a dataset will be generated in the exp_path
    python scripts/tsdf_mapping.py --config active_recon/exp/tsdf.json
    ```
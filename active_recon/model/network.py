import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
import numpy as np


def biased_sigmoid(x, init_bias):
    return torch.sigmoid(x - init_bias)


# Change CutlassMLP to FullyFusedMLP if using advanced(75+) graphics card
class HashNeRF(nn.Module):
    def __init__(
        self,
        scale,
        n_levels=16,
        level_feature=2,
        log2_hash_size=18,
        min_reso=16,
        max_reso=512,
        init_bias=5,
        device=torch.device("cuda"),
    ):
        super(HashNeRF, self).__init__()

        # Obj coord origin at bottom of object
        self.xyz_min = torch.tensor(
            [-scale, -scale, 0 * scale], device=device
        ).unsqueeze(0)
        self.xyz_max = torch.tensor(
            [scale, scale, 2 * scale], device=device
        ).unsqueeze(0)

        self.n_levels = n_levels
        self.level_feature = level_feature
        self.log2_hash_size = log2_hash_size
        self.min_reso = min_reso
        self.max_reso = max_reso
        self.init_bias = init_bias

        self.level_scale = np.exp(
            (np.log(self.max_reso / self.min_reso)) / (self.n_levels - 1)
        )
        
        self.xyz_encoder = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=self.n_levels * self.level_feature + 1,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": self.n_levels,
                "n_features_per_level": self.level_feature,
                "log2_hashmap_size": self.log2_hash_size,
                "base_resolution": self.min_reso,
                "per_level_scale": self.level_scale,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

        self.rgb_net = tcnn.Network(
            n_input_dims=self.n_levels * self.level_feature,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )
        
        self.xyz_encoder_only = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": self.n_levels,
                "n_features_per_level": self.level_feature,
                "log2_hashmap_size": self.log2_hash_size,
                "base_resolution": self.min_reso,
                "per_level_scale": self.level_scale,
            },
        )
        
        # remember your default tensor type
        self.xyz_encoder_pytorch = nn.Sequential(
            self.xyz_encoder_only,
            nn.Linear(self.n_levels * self.level_feature, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_levels * self.level_feature + 1),
        )
        
    def forward(self, x):
        x = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)
        x = torch.clamp(x, 0, 1)
        
        # wether use pytorch or tiny-cuda-nn MLP
        h = self.xyz_encoder(x)
        # h = self.xyz_encoder_pytorch(x)
        
        # wether use biasd initialization
        occs = torch.sigmoid(h[:, 0])
        
        rgbs = self.rgb_net(h[:, 1:])

        return occs.float(), rgbs.float()
    
    def gradient(self, p):
        with torch.enable_grad():
            p.requires_grad_(True)
            y, _ = self.forward(p)
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=p,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True, allow_unused=True)[0]
            return gradients.unsqueeze(1)


class GridNeRF(nn.Module):
    def __init__(
        self,
        scale,
        n_levels=16,
        level_feature=2,
        min_reso=16,
        max_reso=128,
        init_bias=5,
        device=torch.device("cuda"),
    ):
        super(GridNeRF, self).__init__()

        # Obj coord origin at bottom of object
        self.xyz_min = torch.tensor(
            [-scale, -scale, 0 * scale], device=device
        ).unsqueeze(0)
        self.xyz_max = torch.tensor(
            [scale, scale, 2 * scale], device=device
        ).unsqueeze(0)

        self.n_levels = n_levels
        self.level_feature = level_feature
        self.min_reso = min_reso
        self.max_reso = max_reso
        self.init_bias = init_bias

        self.level_scale = np.exp(
            (np.log(self.max_reso / self.min_reso)) / (self.n_levels - 1)
        )

        self.xyz_encoder = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=self.n_levels * self.level_feature + 1,
            encoding_config={
                "otype": "DenseGrid",
                "n_levels": self.n_levels,
                "n_features_per_level": self.level_feature,
                "base_resolution": self.min_reso,
                "per_level_scale": self.level_scale,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )

        self.rgb_net = tcnn.Network(
            n_input_dims=self.n_levels * self.level_feature,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )
        
    def forward(self, x):
        x = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)
        x = torch.clamp(x, 0, 1)
        h = self.xyz_encoder(x)
        occs = torch.sigmoid(h[:, 0])
        rgbs = self.rgb_net(h[:, 1:])

        return occs.float(), rgbs.float()
    
    def gradient(self, p):
        with torch.enable_grad():
            p.requires_grad_(True)
            y, _ = self.forward(p)
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y,
                inputs=p,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True, allow_unused=True)[0]
            return gradients.unsqueeze(1)

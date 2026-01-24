import torch
import torch.nn as nn
from .pointnet_util import PointNetFeaturePropagation, PointNetSetAbstraction2, square_distance, index_points
from .transformer import TransformerBlock


class TransitionDown(nn.Module):
    def __init__(self, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction2(0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)

    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])

    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2


class Backbone(nn.Module):
    def __init__(self, nblocks, nneighbor, d_points, transformer_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(d_points, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1)
            self.transition_downs.append(
                TransitionDown(nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))
        self.nblocks = nblocks

    def forward(self, x):
        xyz = x[..., :3]
        points = self.transformer1(xyz, self.fc1(x[..., 3:]))[0]
        # print('points.shape after the 1st tansformer:')
        # print(points.shape)

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            # print('xyz.shape in transition_downs:')
            # print(xyz.shape)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats

class up_block(nn.Module):
    def __init__(self, up_ratio=4, channels=130, n_c=1):
        super(up_block, self).__init__()
        self.up_ratio = up_ratio

        self.fc = nn.Sequential(
            nn.Linear(channels, 16),
            nn.ReLU(),
            nn.Linear(16, n_c),
            nn.Tanh()
        )
        self.grid = torch.tensor(self.gen_grid(up_ratio)).cuda()

    def forward(self, inputs):
        flow = inputs #b, n, 128
        grid = self.grid.clone()
        grid = grid.unsqueeze(0).repeat(flow.shape[0], 1, flow.shape[1]) #b, 4, 2*n
        grid = grid.view([flow.shape[0], -1, 2]) #b, 4*n, 2

        flow = flow.repeat(1, self.up_ratio, 1) #b, 4n, 128
        flow = torch.cat([flow, grid], dim=2)  #b, n*4, 130
        # flow = flow.permute(0, 2, 1) #b, 130, n*4

        out = self.fc(flow) #b, n*4, 1

        return out


    def gen_grid(self, up_ratio):
        import math
        sqrted = int(math.sqrt(up_ratio))+1
        for i in range(1, sqrted+1).__reversed__():
            if (up_ratio%i) == 0:
                num_x = i
                num_y = up_ratio//i
                break
        grid_x = torch.linspace(-0.2, 0.2, num_x)
        grid_y = torch.linspace(-0.2, 0.2, num_y)

        x, y = torch.meshgrid([grid_x, grid_y])
        grid = torch.stack([x, y], dim=-1) #2, 2, 2
        grid = grid.view([-1, 2]) # 4,2
        return grid

class PointTransformerTriEncoder(nn.Module):
    def __init__(self, nblocks=4, nneighbor=16,  d_points=3, transformer_dim=512):
        super().__init__()
        self.backbone1 = Backbone(nblocks, nneighbor, d_points, transformer_dim)
        self.backbone2 = Backbone(nblocks, nneighbor, d_points, transformer_dim)
        self.backbone3 = Backbone(nblocks, nneighbor, d_points, transformer_dim)

        self.fc1 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 2 ** nblocks)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 2 ** nblocks)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 32 * 2 ** nblocks)
        )
        self.transformer1 = TransformerBlock(32 * 2 ** nblocks, transformer_dim, nneighbor)
        self.transformer2 = TransformerBlock(32 * 2 ** nblocks, transformer_dim, nneighbor)
        self.transformer3 = TransformerBlock(32 * 2 ** nblocks, transformer_dim, nneighbor)



    def forward(self, x):
        # print('x.shape')
        # print(x.shape)
        points, xyz_and_feats = self.backbone1(x)
        xyz = xyz_and_feats[-1][0]
        points = self.transformer1(xyz, self.fc1(points))[0]
        points_gap1 = points.mean(1)  # global average pooling

        points, xyz_and_feats = self.backbone2(x)
        xyz = xyz_and_feats[-1][0]
        points = self.transformer2(xyz, self.fc2(points))[0]
        points_gap2 = points.mean(1)  # global average pooling

        points, xyz_and_feats = self.backbone3(x)
        xyz = xyz_and_feats[-1][0]
        points = self.transformer3(xyz, self.fc3(points))[0]
        points_gap3 = points.mean(1)  # global average pooling


        return points_gap1, points_gap2, points_gap3




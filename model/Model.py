import math
from argparse import Namespace

import torch
import torch.nn as nn
from StackedMamba import StackedMamba

from embeddings.VisionEmbedding import VideoEncoderWithPositionalEncoding


class Model(nn.Module):
    def __init__(self, args: Namespace):
        super(Model, self).__init__()
        self.rgb_embedding = VideoEncoderWithPositionalEncoding(args)
        self.depth_embedding = VideoEncoderWithPositionalEncoding(args)
        self.mamba_blocks = StackedMamba(args)
        # self.positional_encoding = nn.Parameter(torch.randn(args.sequence_length, args.encoder_dim * 2))
        position = torch.arange(0, args.sequence_length, dtype=torch.float).unsqueeze(1)  # (sequence_length, 1)
        div_term = torch.exp(
            torch.arange(0, args.encoder_dim, 2).float() * (-math.log(10000.0) / args.encoder_dim)
        )

        self.positional_encoding = torch.zeros(args.sequence_length, args.encoder_dim, requires_grad=False)
        self.positional_encoding[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(position * div_term)

        self.positional_encoding = self.positional_encoding.to(args.device)

        if args.load_ckpt:
            self.rgb_embedding.load_state_dict(torch.load(f'{args.load_ckpt}/{args.load_ckpt_name}_rgb_final.pth', weights_only=True))
            self.rgb_embedding.eval()
            for param in self.rgb_embedding.parameters():
                param.requires_grad = False
            self.depth_embedding.load_state_dict(torch.load(f'{args.load_ckpt}/{args.load_ckpt_name}_depth_final.pth', weights_only=True))
            self.depth_embedding.eval()
            for param in self.depth_embedding.parameters():
                param.requires_grad = False

            print(f"Load ckpts {f"{args.load_ckpt_name}_rgb_final.pth"} and {f"{args.load_ckpt_name}_depth_final.pth"} Successfully")


    def forward(self, rgb, depth):
        # B, T, C, H, W = rgb.shape
        # rgb_reshaped = rgb.view(B * T, C, H, W)
        # depth_reshaped = depth.view(B * T, C, H, W)
        #
        # rgb = self.rgb_embedding(rgb_reshaped)
        # depth = self.depth_embedding(depth_reshaped)
        #
        # D = rgb.shape[1]
        # rgb_embeddings = rgb.view(B, T, D)
        # depth_embeddings = depth.view(B, T, D)

        B, T, C, H, W = rgb.shape
        rgb = rgb.view(B * T, C, H, W)
        depth = depth.view(B * T, C, H, W)

        rgb_embeddings = self.rgb_embedding(rgb)
        depth_embeddings = self.depth_embedding(depth)

        rgb_embeddings = rgb_embeddings.view(B, T, -1) + self.positional_encoding
        depth_embeddings = depth_embeddings.view(B, T, -1) + self.positional_encoding

        rgbd = torch.cat([rgb_embeddings, depth_embeddings], dim=2)
        # rgbd = self.positional_encoding(rgbd)
        out = self.mamba_blocks(rgbd)
        return out
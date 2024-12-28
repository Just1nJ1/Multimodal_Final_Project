from argparse import Namespace

import timm
import torch
import torch.nn as nn


class PatchEmbedWithPositionalEncoding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # Patch embedding (convolutional projection)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Positional encoding: learnable matrix of size (num_patches, embed_dim)
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

        # Optional normalization layer (Identity by default, as in timm)
        self.norm = nn.Identity()
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize positional encoding (following normal distribution)
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)

    def forward(self, x):
        # Compute patch embeddings
        x = self.proj(x)  # Shape: (B, embed_dim, H_patches, W_patches)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # Shape: (B, num_patches, embed_dim)

        # Add positional encodings
        x = x + self.positional_encoding

        # Apply normalization
        x = self.norm(x)
        return x


class VideoEncoderWithPositionalEncoding(nn.Module):
    def __init__(self, args: Namespace):
        super(VideoEncoderWithPositionalEncoding, self).__init__()
        self.backbone = args.model
        if self.backbone == 'ViT_L':
            ViT = timm.create_model('vit_large_patch16_224', pretrained=True, num_classes=0)
            ViT.blocks = ViT.blocks[:args.blocks]
            self.frame_encoder = ViT
            self.frame_encoder.patch_embed = PatchEmbedWithPositionalEncoding(
                img_size=224, patch_size=16, in_chans=3, embed_dim=1024
            )
            out_features = 1024
        elif self.backbone == 'ViT_Base':
            self.frame_encoder = timm.create_model('vit_base_patch32_224', pretrained=True, num_classes=0)
            self.frame_encoder.patch_embed = PatchEmbedWithPositionalEncoding(
                img_size=224, patch_size=32, in_chans=3, embed_dim=768
            )
            out_features = 768
        elif self.backbone == 'NFNET':
            self.frame_encoder = timm.create_model('nfnet_l0.ra2_in1k', pretrained=True, num_classes=0)
            out_features = 2304
        else:
            raise NotImplementedError(f'Unsupported model: {args.model}')

        self.post_proj = nn.Sequential(
            nn.Linear(out_features, 2 * args.encoder_dim),
            nn.ReLU(),
            nn.Linear(2 * args.encoder_dim, args.encoder_dim),
        )

    def forward(self, x):
        frame_features = self.frame_encoder(x)
        frame_features = self.post_proj(frame_features)
        return frame_features

    def freeze(self):
        for param in self.frame_encoder.parameters():
            param.requires_grad = False

    def unfreeze(self):
        if self.backbone == 'NFNET':
            for param in self.frame_encoder.stages[3].parameters():
                param.requires_grad = True
            for param in self.frame_encoder.final_conv.parameters():
                param.requires_grad = True
        elif self.backbone == 'ViT_L':
            for param in self.frame_encoder.parameters():
                param.requires_grad = True
            # for param in self.frame_encoder.blocks[-1].parameters():
            #     param.requires_grad = True
            # for param in self.frame_encoder.blocks[-2].parameters():
            #     param.requires_grad = True
        elif self.backbone == 'ViT_Base':
            for param in self.frame_encoder.parameters():
                param.requires_grad = True

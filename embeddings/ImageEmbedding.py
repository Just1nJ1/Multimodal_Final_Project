import timm
from torch import nn


class ImageEmbedding(nn.Module):
    def __init__(self, args):
        super(ImageEmbedding, self).__init__()

        assert args.encoder_dim is not None

        self.output_dim = args.encoder_dim

        self.backbone = timm.create_model('nfnet_l0.ra2_in1k', pretrained=True, num_classes=0)
        for param in self.backbone.parameters():
            param.requires_grad = False

        data_config = timm.data.resolve_model_data_config(self.backbone)
        self.backbone_transform = timm.data.create_transform(**data_config, is_training=True)
        self.mlp = nn.Sequential(
            nn.Linear(2304, 1000),
            nn.ReLU(),
            nn.Linear(1000, self.output_dim * 2),
            nn.ReLU(),
            nn.Linear(self.output_dim * 2, self.output_dim),
        )

    def forward(self, x):
        x = self.backbone_transform(x)
        x = self.backbone(x)
        x = self.mlp(x)
        return x

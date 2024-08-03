import torch


class ViTModel(torch.nn.Module):
    pass

class PatchEmbedding(torch.nn.Module):

    def __init__(
            self,
            in_channels: int = 1,
            patch_size: int = 30,
            embedding_dim: int = 900
    ):
        super(PatchEmbedding, self).__init__()

        self.patch_layer = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )

        self.flatten_layer = torch.nn.Flatten(
            start_dim=2,
            end_dim=3
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_layer(x)
        x = self.flatten_layer(x)
        return x.permute(0, 2, 1)  # [B, P^2 * C, N] -> [B, N, P^2 * C]

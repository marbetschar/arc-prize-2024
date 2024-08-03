import math
import torch

class LinearModelV0(torch.nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            out_features: int
    ):
        super(LinearModelV0, self).__init__()

        out_features_sqrt = math.sqrt(in_features)

        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(
            in_features=in_features,
            out_features=hidden_features
        )
        self.linear2 = torch.nn.Linear(
            in_features=hidden_features,
            out_features=out_features
        )
        self.unflatten = torch.nn.Unflatten(
            dim=1,
            unflattened_size=(
                1,
                math.floor(out_features_sqrt),
                math.ceil(out_features_sqrt)
            )
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.unflatten(x)

        return x

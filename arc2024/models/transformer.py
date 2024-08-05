import math
import torch

class TransformerModelV0(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            batch_first: bool = True,
            device: str = 'cpu'
    ):
        super().__init__()

        self.device = device

        d_model_sqrt = math.sqrt(d_model)

        self.flatten = torch.nn.Flatten()
        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            batch_first=batch_first
        )
        self.unflatten = torch.nn.Unflatten(
            dim=1,
            unflattened_size=(
                1,
                math.floor(d_model_sqrt),
                math.ceil(d_model_sqrt)
            )
        )
        self.to(device)


    def forward(self, src, tgt=None):
        src = self.flatten(src)

        if tgt is None:
            x = self.transformer.encoder(src)
            x = self.transformer.decoder(tgt=torch.ones(x.shape).to(self.device), memory=x)
        else:
            tgt = self.flatten(tgt)
            x = self.transformer(src, tgt)

        x = self.unflatten(x)

        return x

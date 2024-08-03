import torch


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true: torch.Tensor, y_pred: torch.Tensor):
    assert y_true.shape == y_pred.shape, "To calculate accuracy of two tensors, they must be of equal shape"

    correct = torch.eq(y_true, y_pred).sum().item()  # torch.eq() calculates where two tensors are equal
    acc = (correct / torch.numel(y_pred)) * 100
    return acc


def norm_arc20204(x: torch.Tensor):
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min)

    return torch.round(torch.round(x_norm * 100) / 10) / 10

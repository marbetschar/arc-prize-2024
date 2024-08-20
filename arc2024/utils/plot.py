import torch
import math
import matplotlib.pyplot as plt

from matplotlib import colors


def as_is(tensor: torch.Tensor):
    # remove batch size and color channel from tensor if necessary (assuming these are first):
    while tensor.ndim > 2:
        tensor = tensor.squeeze(0)

    plt.imshow(tensor.cpu().detach().numpy())


def tensor(
        tensor: torch.Tensor,
        ax=plt,
        title: str = None,
        cmap=None
):
    norm = None
    if cmap is None:
        bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        transparent = (0, 0, 0, 0)
        cmap = colors.ListedColormap(
            ['black', 'gray', 'lightblue', 'blue', 'green', 'yellow', 'orange', 'red', 'darkred', 'pink', transparent]
        )
        norm = colors.BoundaryNorm(bounds, cmap.N)

    # remove batch size and color channel from tensor if necessary (assuming these are first):
    while tensor.ndim > 2:
        tensor = tensor.squeeze(0)

    ax.axis(False)
    if title is not None:
        ax.set_title(title)
    ax.imshow(tensor.cpu().detach().numpy(), cmap=cmap, norm=norm)


def challenge(
        support_set_inputs: [torch.Tensor],
        support_set_outputs: [torch.Tensor],
        query_inputs: [torch.Tensor],
        query_outputs: [torch.Tensor],
        dpi: int = 300
):
    fig, ax = plt.subplots(len(support_set_inputs), 4)
    fig.set_dpi(dpi)

    for i, support_set_input in enumerate(support_set_inputs):
        title = None
        if i == 0:
            title = 'Support Set'
        tensor(support_set_input, ax=ax[i, 0], title=title)
        tensor(support_set_outputs[i], ax=ax[i, 1])

        ax[i, 2].axis(False)
        ax[i, 3].axis(False)

    for i, query_input in enumerate(query_inputs):
        tensor(query_input, ax=ax[0, 2], title='Query')

        if len(query_outputs) > i and query_outputs[i] is not None:
            tensor(query_outputs[i], ax=ax[0, 3])

    plt.show()


def input_and_output(
        X: torch.Tensor,
        y: torch.Tensor,
        y_pred: torch.Tensor = None,
        y_pred_title: str = 'Predicted Output',
        y_pred_cmap = None,
        dpi: int = 300
):
    fig, ax = plt.subplots(1, 3 if y_pred is not None else 2)
    fig.set_dpi(dpi)

    tensor(X, ax=ax[0], title='Input')
    tensor(y, ax=ax[1], title='Correct Output')
    if y_pred is not None:
        tensor(y_pred, ax=ax[2], title=y_pred_title, cmap=y_pred_cmap)

    plt.show()

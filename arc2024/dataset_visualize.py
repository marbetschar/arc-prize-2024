import torch
import math
import matplotlib.pyplot as plt

from matplotlib import colors


def tensor_pad(tensor: torch.Tensor, target_shape=(30, 30), pad_value=10):
    vertical_pad = (target_shape[0] - tensor.shape[0]) / 2.0
    horizontal_pad = (target_shape[1] - tensor.shape[1]) / 2.0

    m = torch.nn.ConstantPad2d(
        padding=(
            math.floor(horizontal_pad),  # padding_left
            math.ceil(horizontal_pad),  # padding_right
            math.floor(vertical_pad),  # padding_top
            math.ceil(vertical_pad)  # padding_bottom
        ),
        value=pad_value
    )

    return m(tensor)


def tensor_plot(tensor: torch.Tensor, ax=plt, title: str = None):
    transparent = (0, 0, 0, 0)
    cmap = colors.ListedColormap(
        ['black', 'gray', 'lightblue', 'blue', 'green', 'yellow', 'orange', 'red', 'darkred', 'pink', transparent])

    bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # remove batch size and color channel from tensor if necessary (assuming these are first):
    while tensor.ndim > 2:
        tensor = tensor.squeeze(0)

    ax.axis(False)
    ax.set_title(title)
    ax.imshow(tensor.detach().numpy(), cmap=cmap, norm=norm)


def challenge_plot(
    challenge_id: str,
    support_set_inputs: [torch.Tensor],
    support_set_outputs: [torch.Tensor],
    query_inputs: [torch.Tensor],
    query_outputs: [torch.Tensor],
    dpi: int = 300
):
    fig, ax = plt.subplots(len(support_set_inputs), 4)

    fig.suptitle(f"Challenge {challenge_id}")
    fig.set_dpi(dpi)

    for i, support_set_input in enumerate(support_set_inputs):
        title = None
        if i == 0:
            title = 'Support Set'
        tensor_plot(tensor_pad(support_set_input), ax=ax[i, 0], title=title)
        tensor_plot(tensor_pad(support_set_outputs[i]), ax=ax[i, 1])

        ax[i, 2].axis(False)
        ax[i, 3].axis(False)

    for i, query_input in enumerate(query_inputs):
        title = None
        if i == 0:
            title = 'Query'
        tensor_plot(tensor_pad(query_input), ax=ax[0, 2], title=title)
        tensor_plot(tensor_pad(query_outputs[i]), ax=ax[0, 3])

    plt.show()

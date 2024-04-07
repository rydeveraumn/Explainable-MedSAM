# stdlib
import math
from typing import List

# third party
import matplotlib.pyplot as plt
import numpy as np


def extract_attention_layers(data: np.lib.npyio.NpzFile) -> List[np.ndarray]:
    """
    From the data outputs we can extract the attention layers.

    TODO: Due to the volume of the data this may not be the most efficient
    but we can modify the data later

    TODO: Also, the usage of 'blocks' may not generalize well, this first
    iteration was created for 2D CT scans
    """
    keys = list(data.keys())
    attention_layers = []
    for key in keys:
        if 'blocks' in key:
            attention_layers.append(data[key])

    return attention_layers


def build_attention_maps(attention_layers):
    """
    Function to build a single plot of attention
    maps

    TODO: Why the 18 in the attention layers?
    Can we do better to compress attention
    mechanisms?
    """
    num_layers = len(attention_layers)

    # Create the grid size for subplots
    grid = int(num_layers * 0.5)
    if grid**2 < num_layers:
        grid += 1

    # Set up the plot
    fig, axes = plt.subplots(grid, grid, figsize=(15, 15))
    axes = axes.flatten()

    for i, attn_map in enumerate(attention_layers):
        # What do these represent?
        B, N, Q, K = attn_map.shape

        # Get the resolution
        res = int(math.sqrt(K))

        v = attn_map[0, 0, 18, :].reshape((res, res))

        # Set up the axis
        ax = axes[i]
        cax = ax.matshow(v, cmap='jet')
        fig.colorbar(cax, ax=ax)

        # Config of the plot
        ax.set_title(f'Layer: {i}', fontsize=10)
        ax.set_xlabel('Keys')
        ax.set_ylabel('Queries')

    # Hide any unused subplots
    for i in range(len(attention_layers), len(axes)):
        axes[i].axis('off')

    fig.tight_layout()
    plt.show()

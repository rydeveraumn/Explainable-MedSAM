# stdlib
import glob
import math
from typing import List

# third party
import cv2
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


def build_attention_maps(attention_layers: List[np.ndarray]):
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

    return fig


def plot_segmentation_masks(modality_type: str = 'CT'):
    """
    Utility function that will plot the segmentation masks
    for different modality types

    TODO: So far this has only been tested on the CT outputs
    from the MedSAM model
    """
    files_lookup = f'/panfs/jay/groups/7/csci5980/senge050/Project/dataset/validation/segs/*{modality_type}*'  # noqa
    files = np.array(glob.glob(files_lookup))

    # We will look at 15 random segmentation masks
    files = np.random.choice(files, 15)

    # first lets plot the segments
    fig, axes = plt.subplots(3, 5, figsize=(10, 6), constrained_layout=True)
    axes = axes.flatten()

    for ax, file in zip(axes, files):
        # Get the segments
        data = np.load(file)
        segment = data['segs']

        # TODO: While looking at the CT scans I noticed that most of the segs
        # have dimension (H, W), however there are some with (C, H, W)
        # This will take the mean along the C dimension but we should try to understand
        # more in depth why these outputs occur
        if len(segment.shape) == 3:
            segment = np.mean(segment, axis=0)

        # Plot the data
        ax.imshow(segment, aspect='auto')
        ax.axis('off')

    fig.suptitle('CT Scan Segmentation Masks')

    return fig


def resize_longest_side(image, target_length=256):
    """
    Resize image to target_length while keeping the aspect ratio
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    oldh, oldw = image.shape[0], image.shape[1]
    scale = target_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)

    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def pad_image(image, target_size=256):
    """
    Pad image to target_size
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    # Pad
    h, w = image.shape[0], image.shape[1]
    padh = np.max(0, target_size - h)
    padw = np.max(0, target_size - w)
    if len(image.shape) == 3:  # Pad image
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else:  # Pad gt mask
        image_padded = np.pad(image, ((0, padh), (0, padw)))

    return image_padded


def get_bbox256(mask_256, bbox_shift=3):
    """
    Get the bounding box coordinates from the mask (256x256)

    Parameters
    ----------
    mask_256 : numpy.ndarray
        the mask of the resized image

    bbox_shift : int
        Add perturbation to the bounding box coordinates

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    y_indices, x_indices = np.where(mask_256 > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates and test the robustness
    # this can be removed if you do not want to test the robustness
    H, W = mask_256.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)

    bboxes256 = np.array([x_min, y_min, x_max, y_max])

    return bboxes256


def resize_box_to_256(box, original_size):
    """
    the input bounding box is obtained from the original image
    here, we rescale it to the coordinates of the resized image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    original_size : tuple
        the original size of the image

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    new_box = np.zeros_like(box)
    ratio = 256 / max(original_size)
    for i in range(len(box)):
        new_box[i] = int(box[i] * ratio)

    return new_box

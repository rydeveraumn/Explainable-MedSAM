# stdlib
import glob
import os
from collections import OrderedDict
from datetime import datetime
from time import time
from typing import List

# third party
import click
import numpy as np
import pandas as pd
import torch
import tqdm
import wandb
from matplotlib import pyplot as plt
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from segment_anything.modeling.transformer import Attention as SamAttention
from skimage import transform
from torchvision.ops import masks_to_boxes

# first party
from tiny_vit_sam import Attention as ViTAttention
from tiny_vit_sam import TinyViT
from xai_medsam.metrics import compute_multi_class_dsc
from xai_medsam.models import MedSAM_Lite

# Training data path
TRAIN_DATA_PATH = '/panfs/jay/groups/7/csci5980/senge050/Project/dataset/train_npz'
VALIDATION_DATA_PATH = '/panfs/jay/groups/7/csci5980/dever120/Explainable-MedSam/datasets/validation-v2/'  # noqa
SAVE_DATA_PATH = VALIDATION_DATA_PATH  # noqa
PRED_SAVE_DIR = '/panfs/jay/groups/7/csci5980/dever120/Explainable-MedSam/datasets/validation-medsam-lite-segs-v3/'  # noqa
# PRED_SAVE_DIR = '/panfs/jay/groups/7/csci5980/dever120/Explainable-MedSam/datasets/validation-medsam-lite-segs/'  # noqa

# Modalities in the training data
MODALITIES = [
    'Dermoscopy',
    'Endoscopy',
    'Fundus',
    'Mammography',
    'Microscopy',
    'OCT',
    'PET',
    'US',
    'XRay',
]
SAMPLES = 250


# Add click group for cli commands
@click.group()
def cli():  # noqa
    pass


def test():
    print('Hello World')


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_256, new_size, original_size):
    """
    Perform inference using the LiteMedSAM model.

    Args:
        medsam_model (MedSAMModel): The MedSAM model.
        img_embed (torch.Tensor): The image embeddings.
        box_256 (numpy.ndarray): The bounding box coordinates.
        new_size (tuple): The new size of the image.
        original_size (tuple): The original size of the image.
    Returns:
        tuple: A tuple containing the segmented image and
        the intersection over union (IoU) score.
    """
    box_torch = torch.as_tensor(
        box_256[None, None, ...], dtype=torch.float, device=img_embed.device
    )

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, iou = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = medsam_model.postprocess_masks(
        low_res_logits, new_size, original_size
    )
    low_res_pred = torch.sigmoid(low_res_pred)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

    return medsam_seg, iou


def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """
    show mask on the image

    Parameters
    ----------
    mask : numpy.ndarray
        mask of the image
    ax : matplotlib.axes.Axes
        axes to plot the mask
    mask_color : numpy.ndarray
        color of the mask
    alpha : float
        transparency of the mask
    """
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, edgecolor='blue'):
    """
    show bounding box on the image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    ax : matplotlib.axes.Axes
        axes to plot the bounding box
    edgecolor : str
        color of the bounding box
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0, 0, 0, 0), lw=2)
    )


def get_attns(module, prefix=''):
    """
    Function to extract output of attention
    layers
    """
    attns = {}
    for name, m in module.named_modules():
        if isinstance(m, SamAttention) or isinstance(m, ViTAttention):
            attns[prefix + name] = m.attention_map.cpu().numpy().astype(np.float16)

    return attns


def MedSAM_infer_npz_2D(
    img_npz_file,
    pred_save_dir,
    medsam_lite_model,
    device,
    attention=False,
    png_save_dir=None,
):
    npz_name = os.path.basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)  # (H, W, 3)
    img_3c = npz_data['imgs']  # (H, W, 3)

    if len(img_3c.shape) < 3:
        # Need to create a 3D image
        # stack along the last axis
        img_3c = np.stack([img_3c] * 3, axis=-1)
    assert (
        np.max(img_3c) < 256
    ), f'input data should be in range [0, 255], but got {np.unique(img_3c)}'  # noqa
    H, W, _ = img_3c.shape
    boxes = npz_data['boxes']
    segs = np.zeros((H, W), dtype=np.uint8)

    # preprocessing
    # This comes from the tutorial and seems to yield better results
    # for the bounding box resize
    target_size = 256
    img_256 = transform.resize(
        img_3c,
        (target_size, target_size),
        order=3,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.uint8)

    newh, neww, _ = img_256.shape
    img_256_norm = (img_256 - img_256.min()) / np.clip(
        img_256.max() - img_256.min(), a_min=1e-8, a_max=None
    )
    img_256_tensor = (
        torch.tensor(img_256_norm).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )
    attns = {}
    with torch.no_grad():
        image_embedding = medsam_lite_model.image_encoder(img_256_tensor)
        attns.update(get_attns(medsam_lite_model.image_encoder))

    for idx, box in enumerate(boxes, start=1):
        box256 = box / np.array([W, H, W, H]) * target_size
        box256 = box256[None, ...]  # (1, 4)
        medsam_mask, iou_pred = medsam_inference(
            medsam_lite_model, image_embedding, box256, (newh, neww), (H, W)
        )
        attns.update(get_attns(medsam_lite_model.mask_decoder, prefix=f'box{idx}_'))
        segs[medsam_mask > 0] = idx
        # print(f'{npz_name}, box: {box},
        # predicted iou: {np.round(iou_pred.item(), 4)}')

    to_save = {'segs': segs, **(attns if attention else {})}
    np.savez_compressed(
        os.path.join(pred_save_dir, npz_name),
        **to_save,
    )
    if png_save_dir is not None:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3c)
        ax[1].imshow(img_3c)
        ax[0].set_title("Image")
        ax[1].set_title("LiteMedSAM Segmentation")
        ax[0].axis('off')
        ax[1].axis('off')

        for i, box in enumerate(boxes):
            color = np.random.rand(3)
            box_viz = box
            show_box(box_viz, ax[1], edgecolor=color)
            show_mask((segs == i + 1).astype(np.uint8), ax[1], mask_color=color)

        plt.tight_layout()
        os.makedirs(png_save_dir, exist_ok=True)
        plt.savefig(
            os.path.join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300
        )
        plt.close()


def add_multiple_image_validation_data_from_train(modality: str = 'CT') -> None:
    """
    CT/MR data is set up different from other modality types
    so we created a method to handle CT data differently
    """
    # Get random files from CT
    path = os.path.join(TRAIN_DATA_PATH, f'{modality}/*/*')

    # Glob will give us a list of all of the files in the directory
    files = glob.glob(path)
    files = list(np.random.choice(files, SAMPLES))

    # Iterate over the different files
    for idx, file in enumerate(files):
        # get the file name
        name = file.split('/')[-1]

        # Load in the data
        data = np.load(file, 'r', allow_pickle=True)

        # Just pick the first image
        img = data['imgs'][0, :, :]
        gts = data['gts'][0, :, :]

        # Create bounding box from masks
        mask = torch.tensor(gts.astype(np.int32))

        # We get the unique colors, as these would be the object ids.
        obj_ids = torch.unique(mask)

        # first id is the background, so remove it.
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set of boolean masks.
        # Note that this snippet would work as well
        # if the masks were float values instead of ints.
        masks = mask == obj_ids[:, None, None]

        # Create the boxes with torchvision module
        boxes = masks_to_boxes(masks)

        # Save the data
        file_save_path = os.path.join(SAVE_DATA_PATH, name)
        np.savez(file_save_path, imgs=img, gts=mask, boxes=boxes)


def build_validation_data_from_train() -> None:
    """
    This information comes from the dataset email:

    The ground truth of the validation set will not be released.
    You can obtain the metrics (DSC and NSD scores) by submitting
    segmentation results on the Codabench platform.

    The validation set contains 9 modalities, which is only a
    small proportion of the testing set. In other words,
    the role of the validation set is the sanity check of the algorithms,
    which cannot reflect the complete performance on the hidden testing set.
    We recommend building your validation set as well by
    selecting 5-10% of the training cases.
    """
    files = []
    for modality in tqdm.tqdm(MODALITIES):
        # Load in data from modality
        path = os.path.join(TRAIN_DATA_PATH, f'{modality}/*/*')

        # Glob will give us a list of all of the files in the directory
        mod_files = glob.glob(path)
        mod_files = list(np.random.choice(mod_files, SAMPLES))
        files.extend(mod_files)

    # Iterate over the files and omit any images that are not 2D
    # based off a simple heuristic
    for idx, file in enumerate(files):
        name = file.split('/')[-1]
        # Load in the data
        data = np.load(file)
        img = data['imgs']

        # If this condition is true then add to the validation set
        if len(img.shape) == 3 and (img.shape[2] == 3):
            # Create bounding box from masks
            mask = torch.tensor(data['gts'].astype(np.int32))

            # We get the unique colors, as these would be the object ids.
            obj_ids = torch.unique(mask)

            # first id is the background, so remove it.
            obj_ids = obj_ids[1:]

            # split the color-encoded mask into a set of boolean masks.
            # Note that this snippet would work as well
            # if the masks were float values instead of ints.
            masks = mask == obj_ids[:, None, None]

            # Create the boxes with torchvision module
            boxes = masks_to_boxes(masks)

            # Save the data
            file_save_path = os.path.join(SAVE_DATA_PATH, name)
            np.savez(file_save_path, imgs=img, gts=mask, boxes=boxes)

    # Add CT data separately
    add_multiple_image_validation_data_from_train(modality='CT')

    # Add MR data separately
    add_multiple_image_validation_data_from_train(modality='MR')


@cli.command('run-inference')
@click.option('-i', '--input_dir', type=str, help='Root directory of the data')
@click.option('-o', '--output_dir', type=str, help='Directory to save the prediction')
@click.option(
    '--lite_medsam_checkpoint_path',
    type=str,
    help='Path to the checkpoint of MedSAM-Lite',
)
@click.option(
    '-d', '--device', type=str, default='cpu', help='Device to run the inference'
)
@click.option(
    '--png_save_dir', type=str, default=None, help='Directory to save the overlay image'
)
@click.option('--attention', is_flag=True, default=False, help='Save attention scores')
@click.option(
    '--samples', type=int, default=0, help='Max number of samples to run inference on'
)
def run_inference(
    input_dir,
    output_dir,
    lite_medsam_checkpoint_path,
    device,
    png_save_dir,
    attention,
    samples,
) -> None:
    """
    Task to run inference on validation images. This will save the
    segmentation masks so we can compute metrics.

    TODO: Add Corey's attention extraction
    """
    # Set seeds
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(2024)
    torch.cuda.manual_seed(2024)
    np.random.seed(2024)

    API_KEY = '2080070c4753d0384b073105ed75e1f46669e4bf'
    PROJECT_NAME = 'Explainable-MedSAM'

    # Enable wandb
    wandb.login(key=API_KEY)

    # Initalize wandb
    # TODO: Save training and validation curves per fold
    wandb.init(
        # set the wandb project where this run will be logged
        project=PROJECT_NAME,
        tags=['xai-medsam'],
    )
    print('Start inference 🎉')

    # Device will be cpu
    device = torch.device('cpu')

    # Set up the image MedSAM image encoder
    medsam_lite_image_encoder = TinyViT(
        img_size=256,
        in_chans=3,
        embed_dims=[
            64,  # (64, 256, 256)
            128,  # (128, 128, 128)
            160,  # (160, 64, 64)
            320,  # (320, 64, 64)
        ],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8,
    )

    # Set up the prompt encoder
    medsam_lite_prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16,
    )

    # Set up the mask encoder
    medsam_lite_mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )

    medsam_lite_model = MedSAM_Lite(
        image_encoder=medsam_lite_image_encoder,
        mask_decoder=medsam_lite_mask_decoder,
        prompt_encoder=medsam_lite_prompt_encoder,
    )

    # For this code the model path will be fixed
    lite_medsam_checkpoint = torch.load(
        lite_medsam_checkpoint_path,
        map_location='cpu',
    )
    medsam_lite_model.load_state_dict(lite_medsam_checkpoint)
    medsam_lite_model.to(device)
    medsam_lite_model.eval()

    # Iterate over the saved data
    validation_files = glob.glob(os.path.join(input_dir, '*.npz'), recursive=True)
    if samples > 0:
        r = np.random.default_rng(0)
        validation_files = r.choice(validation_files, samples, replace=False)

    efficiency = OrderedDict()  # type: ignore
    efficiency['case'] = []
    efficiency['time'] = []

    # Run the inference for all validation data
    exceptions_list: List[str] = []
    pbar = tqdm.tqdm(validation_files)
    for img_npz_file in pbar:
        start_time = time()
        try:
            MedSAM_infer_npz_2D(
                img_npz_file=img_npz_file,
                pred_save_dir=output_dir,
                medsam_lite_model=medsam_lite_model,
                device=device,
                attention=attention,
                png_save_dir=png_save_dir,
            )

        except Exception as e:
            pbar.write(f'Error in file {img_npz_file}: {e}')
            exceptions_list.append(img_npz_file)

        end_time = time()
        efficiency['case'].append(os.path.basename(img_npz_file))
        efficiency['time'].append(end_time - start_time)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pbar.write(
            f'{current_time} file name: {os.path.basename(img_npz_file)}'
            f' time cost: {np.round(end_time - start_time, 4)}'
        )
    efficiency_df = pd.DataFrame(efficiency)
    efficiency_df.to_csv(os.path.join(output_dir, 'efficiency.csv'), index=False)

    print('Inference completed! ✅')


def compute_metrics(save_version: str = 'v1') -> None:
    """
    Task to compute the Dice coefficient
    """
    validation_data = glob.glob(os.path.join(VALIDATION_DATA_PATH, '*'))
    validation_file_names = []
    for name in validation_data:
        file = name.split('/')[-1]
        validation_file_names.append(file)

    metrics = []
    for name in tqdm.tqdm(validation_file_names):
        img_npz_file = os.path.join(VALIDATION_DATA_PATH, name)
        pred_path = os.path.join(PRED_SAVE_DIR, name)
        modality_type = name.split('-')[0]

        # Load the data & compute metrics for image
        npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
        gt = npz_data['gts']

        # Weird issue?
        # Just compute as one large segmentation
        gt[gt != 0] = 1

        # Load predictions
        predictions = np.load(pred_path, 'r', allow_pickle=True)
        segs = predictions['segs']

        # Need to do the same with the predictions
        segs[segs != 0] = 1

        # Dice score
        dsc_score = compute_multi_class_dsc(gt, segs)

        metrics.append((name, modality_type, dsc_score))

    # Turn metrics into pandas data frame
    metrics_df = pd.DataFrame(
        metrics, columns=['filename', 'modality_type', 'dice_score']
    )
    metrics_df.to_parquet(
        os.path.join(
            '/panfs/jay/groups/7/csci5980/dever120/Explainable-MedSam/datasets',
            f'medsam-metrics-{save_version}.parquet',
        )
    )


if __name__ == "__main__":
    # Be able to run different commands
    cli()

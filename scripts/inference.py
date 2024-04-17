import os
from glob import glob
from tqdm import tqdm
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT
from matplotlib import pyplot as plt
import cv2
import argparse
from collections import OrderedDict
import pandas as pd
from datetime import datetime
from xai_medsam import MedSAM_Lite

import random
from tiny_vit_sam import Attention as ViTAttention
from segment_anything.modeling.transformer import Attention as SamAttention


#%% set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

parser = argparse.ArgumentParser()

parser.add_argument(
    '-i',
    '--input_dir',
    type=str,
    default='test_demo/imgs/',
    # required=True,
    help='root directory of the data',
)
parser.add_argument(
    '-o',
    '--output_dir',
    type=str,
    default='test_demo/segs/',
    help='directory to save the prediction',
)
parser.add_argument(
    '--lite_medsam_checkpoint_path',
    type=str,
    default="work_dir/LiteMedSAM/lite_medsam.pth",
    help='path to the checkpoint of MedSAM-Lite',
)
parser.add_argument(
    '-d',
    '--device',
    type=str,
    default="cpu",
    help='device to run the inference',
)
parser.add_argument(
    '--save_overlay',
    default=True,
    action='store_true',
    help='whether to save the overlay image'
)
parser.add_argument(
    '--png_save_dir',
    type=str,
    default='./overlay',
    help='directory to save the overlay image'
)
parser.add_argument(
    '--attention',
    default=False,
    action='store_true',
    help='save attention scores'
)
parser.add_argument(
    '--samples',
    type=int,
    default=0,
    help='Max number of samples to run inference on'
)

args = parser.parse_args()

data_root = args.input_dir
pred_save_dir = args.output_dir
save_overlay = args.save_overlay
if save_overlay:
    assert args.png_save_dir is not None, "Please specify the directory to save the overlay image"
    png_save_dir = args.png_save_dir
    os.makedirs(png_save_dir, exist_ok=True)

lite_medsam_checkpoint_path = args.lite_medsam_checkpoint_path
os.makedirs(pred_save_dir, exist_ok=True)
device = torch.device(args.device)
image_size = 256

from xai_medsam.utils import resize_longest_side, pad_image
from xai_medsam import MedSAM_Lite


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
        color = np.array([251/255, 252/255, 30/255, alpha])
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
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))     

from xai_medsam.utils import get_bbox256, resize_box_to_256
from xai_medsam.tasks import medsam_inference


medsam_lite_image_encoder = TinyViT(
    img_size=256,
    # in_chans=3,
    embed_dims=[
        64, ## (64, 256, 256)
        128, ## (128, 128, 128)
        160, ## (160, 64, 64)
        320 ## (320, 64, 64) 
    ],
    depths=[2, 2, 6, 2],
    num_heads=[2, 4, 5, 10],
    window_sizes=[7, 7, 14, 7],
    mlp_ratio=4.,
    drop_rate=0.,
    drop_path_rate=0.0,
    use_checkpoint=False,
    mbconv_expand_ratio=4.0,
    local_conv_size=3,
    layer_lr_decay=0.8
)

medsam_lite_prompt_encoder = PromptEncoder(
    embed_dim=256,
    image_embedding_size=(64, 64),
    input_image_size=(256, 256),
    mask_in_chans=16
)

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
    image_encoder = medsam_lite_image_encoder,
    mask_decoder = medsam_lite_mask_decoder,
    prompt_encoder = medsam_lite_prompt_encoder
)

lite_medsam_checkpoint = torch.load(lite_medsam_checkpoint_path, map_location='cpu')
medsam_lite_model.load_state_dict(lite_medsam_checkpoint)
medsam_lite_model.to(device)
medsam_lite_model.eval()

def get_attns(module, prefix=''):
    attns = {}
    for name, m in module.named_modules():
        if isinstance(m, SamAttention) or isinstance(m, ViTAttention):
            attns[prefix + name] = m.attention_map.cpu().numpy()
    return attns

# def MedSAM_infer_npz_2D(img_npz_file, attention=False):
#     npz_name = os.path.basename(img_npz_file)
#     npz_data = np.load(img_npz_file, 'r', allow_pickle=True) # (H, W, 3)
#     img_3c = npz_data['imgs'] # (H, W, 3)
#     assert np.max(img_3c)<256, f'input data should be in range [0, 255], but got {np.unique(img_3c)}'
#     H, W = img_3c.shape[:2]
#     boxes = npz_data['boxes']
#     segs = np.zeros(img_3c.shape[:2], dtype=np.uint8)

#     ## preprocessing
#     img_256 = resize_longest_side(img_3c, 256)
#     newh, neww = img_256.shape[:2]
#     img_256_norm = (img_256 - img_256.min()) / np.clip(
#         img_256.max() - img_256.min(), a_min=1e-8, a_max=None
#     )
#     img_256_padded = pad_image(img_256_norm, 256)
#     img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
#     attns = {}
#     with torch.no_grad():
#         image_embedding = medsam_lite_model.image_encoder(img_256_tensor)
#         attns.update(get_attns(medsam_lite_model.image_encoder))

#     for idx, box in enumerate(boxes, start=1):
#         box256 = resize_box_to_256(box, original_size=(H, W))
#         box256 = box256[None, ...] # (1, 4)
#         medsam_mask, iou_pred = medsam_inference(medsam_lite_model, image_embedding, box256, (newh, neww), (H, W))
#         attns.update(get_attns(medsam_lite_model.mask_decoder, prefix=f"box{idx}_"))
#         segs[medsam_mask>0] = idx
#         # print(f'{npz_name}, box: {box}, predicted iou: {np.round(iou_pred.item(), 4)}')

#     to_save = {
#         'segs': segs,
#         **(attns if attention else {})
#     }
#     np.savez_compressed(
#         os.path.join(pred_save_dir, npz_name),
#         **to_save,
#     )

#     # visualize image, mask and bounding box
#     if save_overlay:
#         fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#         ax[0].imshow(img_3c)
#         ax[1].imshow(img_3c)
#         ax[0].set_title("Image")
#         ax[1].set_title("LiteMedSAM Segmentation")
#         ax[0].axis('off')
#         ax[1].axis('off')

#         for i, box in enumerate(boxes):
#             color = np.random.rand(3)
#             box_viz = box
#             show_box(box_viz, ax[1], edgecolor=color)
#             show_mask((segs == i+1).astype(np.uint8), ax[1], mask_color=color)

#         plt.tight_layout()
#         plt.savefig(os.path.join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
#         plt.close()

from xai_medsam.tasks import MedSAM_infer_npz_2D

def MedSAM_infer_npz_3D(img_npz_file, attention=False):
    npz_name = os.path.basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
    img_3D = npz_data['imgs'] # (D, H, W)
    spacing = npz_data['spacing'] # not used in this demo because it treats each slice independently
    segs = np.zeros_like(img_3D, dtype=np.uint8) 
    boxes_3D = npz_data['boxes'] # [[x_min, y_min, z_min, x_max, y_max, z_max]]

    for idx, box3D in enumerate(boxes_3D, start=1):
        segs_3d_temp = np.zeros_like(img_3D, dtype=np.uint8) 
        x_min, y_min, z_min, x_max, y_max, z_max = box3D
        assert z_min < z_max, f"z_min should be smaller than z_max, but got {z_min=} and {z_max=}"
        mid_slice_bbox_2d = np.array([x_min, y_min, x_max, y_max])
        z_middle = int((z_max - z_min)/2 + z_min)

        # infer from middle slice to the z_max
        # print(npz_name, 'infer from middle slice to the z_max')
        for z in range(z_middle, z_max):
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape

            img_256 = resize_longest_side(img_3c, 256)
            new_H, new_W = img_256.shape[:2]

            img_256 = (img_256 - img_256.min()) / np.clip(
                img_256.max() - img_256.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            ## Pad image to 256x256
            img_256 = pad_image(img_256)
            
            # convert the shape to (3, H, W)
            img_256_tensor = torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0).to(device)
            # get the image embedding
            with torch.no_grad():
                image_embedding = medsam_lite_model.image_encoder(img_256_tensor) # (1, 256, 64, 64)
            if z == z_middle:
                box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
            else:
                pre_seg = segs[z-1, :, :]
                if np.max(pre_seg) > 0:
                    pre_seg256 = resize_longest_side(pre_seg)
                    pre_seg256 = pad_image(pre_seg256)
                    box_256 = get_bbox256(pre_seg256)
                else:
                    box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
            img_2d_seg, iou_pred = medsam_inference(medsam_lite_model, image_embedding, box_256, [new_H, new_W], [H, W])
            segs_3d_temp[z, img_2d_seg>0] = idx
        
        # infer from middle slice to the z_max
        # print(npz_name, 'infer from middle slice to the z_min')
        for z in range(z_middle-1, z_min, -1):
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape

            img_256 = resize_longest_side(img_3c)
            new_H, new_W = img_256.shape[:2]

            img_256 = (img_256 - img_256.min()) / np.clip(
                img_256.max() - img_256.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            ## Pad image to 256x256
            img_256 = pad_image(img_256)

            img_256_tensor = torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0).to(device)
            # get the image embedding
            with torch.no_grad():
                image_embedding = medsam_lite_model.image_encoder(img_256_tensor) # (1, 256, 64, 64)

            pre_seg = segs[z+1, :, :]
            if np.max(pre_seg) > 0:
                pre_seg256 = resize_longest_side(pre_seg)
                pre_seg256 = pad_image(pre_seg256)
                box_256 = get_bbox256(pre_seg256)
            else:
                scale_256 = 256 / max(H, W)
                box_256 = mid_slice_bbox_2d * scale_256
            img_2d_seg, iou_pred = medsam_inference(medsam_lite_model, image_embedding, box_256, [new_H, new_W], [H, W])
            segs_3d_temp[z, img_2d_seg>0] = idx
        segs[segs_3d_temp>0] = idx
    np.savez_compressed(
        os.path.join(pred_save_dir, npz_name),
        segs=segs,
    )            

    # visualize image, mask and bounding box
    if save_overlay:
        idx = int(segs.shape[0] / 2)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3D[idx], cmap='gray')
        ax[1].imshow(img_3D[idx], cmap='gray')
        ax[0].set_title("Image")
        ax[1].set_title("LiteMedSAM Segmentation")
        ax[0].axis('off')
        ax[1].axis('off')

        for i, box3D in enumerate(boxes_3D, start=1):
            if np.sum(segs[idx]==i) > 0:
                color = np.random.rand(3)
                x_min, y_min, z_min, x_max, y_max, z_max = box3D
                box_viz = np.array([x_min, y_min, x_max, y_max])
                show_box(box_viz, ax[1], edgecolor=color)
                show_mask(segs[idx]==i, ax[1], mask_color=color)

        plt.tight_layout()
        plt.savefig(os.path.join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
        plt.close()


if __name__ == '__main__':
    print("running ml4h modified, attention:", args.attention)
    img_npz_files = sorted(glob(os.path.join(data_root, '*.npz'), recursive=True))
    if args.samples > 0:
        random.seed(0)
        img_npz_files = random.sample(img_npz_files, args.samples)
    efficiency = OrderedDict()
    efficiency['case'] = []
    efficiency['time'] = []
    for img_npz_file in tqdm(img_npz_files):
        start_time = time()
        if os.path.basename(img_npz_file).startswith('3D'):
            MedSAM_infer_npz_3D(img_npz_file, args.attention)
        else:
            MedSAM_infer_npz_2D(img_npz_file, pred_save_dir, medsam_lite_model, device, args.attention, save_overlay, png_save_dir)
        end_time = time()
        efficiency['case'].append(os.path.basename(img_npz_file))
        efficiency['time'].append(end_time - start_time)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(current_time, 'file name:', os.path.basename(img_npz_file), 'time cost:', np.round(end_time - start_time, 4))
    efficiency_df = pd.DataFrame(efficiency)
    efficiency_df.to_csv(os.path.join(pred_save_dir, 'efficiency.csv'), index=False)

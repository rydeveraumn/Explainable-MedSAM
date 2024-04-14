# stdlib
import glob
import os

# third party
import numpy as np
import torch
import tqdm
from torchvision.ops import masks_to_boxes

# first party
from xai_medsam.utils import pad_image, resize_box_to_256, resize_longest_side

# Training data path
TRAIN_DATA_PATH = '/panfs/jay/groups/7/csci5980/senge050/Project/dataset/train_npz'
SAVE_DATA_PATH = '/panfs/jay/groups/7/csci5980/dever120/Explainable-MedSam/datasets/validation'  # noqa

# Modalities in the training data
MODALITIES = [
    'CT',
    'Dermoscopy',
    'Endoscopy',
    'Fundus',
    'Mammography',
    'Microscopy',
    'MR',
    'OCT',
    'PET',
    'US',
    'XRay',
]


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


def MedSAM_infer_npz_2D(img_npz_file, pred_save_dir, medsam_lite_model, device):
    npz_name = img_npz_file.split('/')[-1]
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)  # (H, W, 3)
    img_3c = npz_data['imgs']  # (H, W, 3)
    assert (
        np.max(img_3c) < 256
    ), f'input data should be in range [0, 255], but got {np.unique(img_3c)}'  # noqa
    H, W, _ = img_3c.shape
    boxes = npz_data['boxes']
    segs = np.zeros(img_3c.shape[:2], dtype=np.uint8)

    # preprocessing
    img_256 = resize_longest_side(img_3c, 256)
    newh, neww = img_256.shape[:2]
    img_256_norm = (img_256 - img_256.min()) / np.clip(
        img_256.max() - img_256.min(), a_min=1e-8, a_max=None
    )
    img_256_padded = pad_image(img_256_norm, 256)
    img_256_tensor = (
        torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )
    with torch.no_grad():
        image_embedding = medsam_lite_model.image_encoder(img_256_tensor)

    for idx, box in enumerate(boxes, start=1):
        box256 = resize_box_to_256(box, original_size=(H, W))
        box256 = box256[None, ...]  # (1, 4)
        medsam_mask, iou_pred = medsam_inference(
            medsam_lite_model, image_embedding, box256, (newh, neww), (H, W)
        )
        segs[medsam_mask > 0] = idx
        # print(f'{npz_name}, box: {box},
        # predicted iou: {np.round(iou_pred.item(), 4)}')

    np.savez_compressed(
        os.path.join(pred_save_dir, npz_name),
        segs=segs,
    )


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
    for modality in tqdm.tqdm(MODALITIES):
        # Load in data from modality
        path = os.path.join(TRAIN_DATA_PATH, f'{modality}/*/*')

        # Glob will give us a list of all of the files in the directory
        files = glob.glob(path)
        files = list(np.random.choice(files, 100))

    # Iterate over the files and omit any images that are not 2D
    # based off a simple heuristic
    for idx, file in enumerate(files):
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
            file_save_path = os.path.join(SAVE_DATA_PATH, f'{modality}-{idx}.npz')
            np.savez(file_save_path, imgs=img, gts=mask, boxes=boxes)

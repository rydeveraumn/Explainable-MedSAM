# third party
from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer

# first party
from tiny_vit_sam import TinyViT
from skimage import transform


class MedSAM_Lite(nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def forward(self,
                image: torch.Tensor,
                box: Union[torch.Tensor, np.ndarray],
                new_size: Tuple[int, int],
                original_size: Tuple[int, int],
                raw_predictions = False) -> Tuple[torch.Tensor, torch.Tensor]:
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)

        # do not compute gradients for prompt encoder
        with torch.no_grad():
            if isinstance(box, np.ndarray):
                box = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box.shape) == 2:
                box = box[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box,
                masks=None,
            )
        
        low_res_logits, iou = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )  # (B, 1, 256, 256)
        
        if raw_predictions:
            return low_res_logits

        low_res_pred = self.postprocess_masks(
            low_res_logits, new_size, original_size
        )
        low_res_pred = torch.sigmoid(low_res_pred)
        return low_res_pred, iou

    @torch.no_grad()
    def postprocess_masks(self, masks, new_size, original_size):
        """
        Do cropping and resizing

        Parameters
        ----------
        masks : torch.Tensor
            masks predicted by the model
        new_size : tuple
            the shape of the image after resizing to the longest side of 256
        original_size : tuple
            the original shape of the image

        Returns
        -------
        torch.Tensor
            the upsampled mask to the original size
        """
        # Crop
        masks = masks[..., : new_size[0], : new_size[1]]
        # Resize
        masks = F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )

        return masks
    
    def preprocess_2d_img(self, img: np.ndarray, target_size: int = 256) -> torch.Tensor:
        if len(img.shape) < 3:
            # Need to create a 3D image
            # stack along the last axis
            img = np.stack([img] * 3, axis=-1)
        assert np.max(img) < 256, f'input data should be in range [0, 255], but got {np.unique(img)}'
        
        # preprocessing
        # This comes from the tutorial and seems to yield better results
        # for the bounding box resize
        img = transform.resize(img, (target_size, target_size), order=3, preserve_range=True, anti_aliasing=True)
        img = img.astype(np.uint8)
        img = (img - img.min()) / np.clip(img.max() - img.min(), a_min=1e-8, a_max=None)
        img_t = torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0)
        return img_t
    
    @classmethod
    def from_medsam_lite(cls, checkpoint):
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
            checkpoint,
            map_location='cpu',
        )
        medsam_lite_model.load_state_dict(lite_medsam_checkpoint)
        return medsam_lite_model
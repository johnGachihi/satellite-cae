import torch
import rasterio
import numpy as np
from torchvision import transforms
from models.modeling_cae import cae_base_patch16_224_8k_vocab
from furnace.dataset_folder import SentinelNormalize
from furnace.masking_generator import MaskingGenerator
from timm.models import create_model


def patchify(imgs, c, p=16):
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
    return x

def unpatchify(x, c, p=16):
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
        
    x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
    return imgs

def transform(img):
    # Normalize
    mean = (1370.19151926, 1184.3824625 , 1120.77120066, 1136.26026392,
            1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
            1972.62420416,  582.72633433,   14.77112979, 1732.16362238, 1247.91870117)
    std = (633.15169573,  650.2842772 ,  712.12507725,  965.23119807,
           948.9819932 , 1108.06650639, 1258.36394548, 1233.1492281 ,
           1364.38688993,  472.37967789,   14.3114637 , 1310.36996126, 1087.6020813)

    mean, std = np.array(mean), np.array(std)
    min_value = (mean - 2 * std).reshape(-1, 1, 1)
    max_value = (mean + 2 * std).reshape(-1, 1, 1)
    img = (img - min_value) / (max_value - min_value) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    img = transforms.ToTensor()(img)
    img = img.permute(1, 2, 0)
    img = transforms.Resize(
        (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
    )(img)

    return img

def save_tiff(name, data, meta):
    f_name = f"/home/ubuntu/satellite-cae/CAE/output/sample_tiffs/{name}.tiff"

    with rasterio.open(f_name, "w", **meta) as dst:
        dst.write(data)


if __name__ == "__main__":
    # Read image
    tiff_path = "/home/ubuntu/satellite-cae/SatMAE/data/fmow-sentinel/train/airport/airport_0/airport_0_100.tif"
    with rasterio.open(tiff_path) as src:
        img = src.read()
        tiff_meta = src.meta.copy()

    print(f"original img {img.shape}")

    orig_img = transforms.ToTensor()(img).float()
    orig_img = orig_img.permute(1, 2, 0)
    orig_img = transforms.Resize(
        (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
    )(orig_img)
    save_tiff("orig_img", orig_img, tiff_meta)
    
    img = transform(img)
    save_tiff("transformed_img", img, tiff_meta)
    
    print(f"transformed {img.shape}")

    img = img[None, ...] # Add batch dim    

    patches = patchify(img, c=13)
    print(f"Patchified {patches.shape}")

    # Create mask
    masked_position_generator = MaskingGenerator(
        (14, 14), num_masking_patches=98,
        max_num_patches=98,
        min_num_patches=98,
    )

    mask = masked_position_generator()[None, ...]
    mask = torch.from_numpy(mask)
    mask = mask.flatten(1).to(torch.bool)
    print(f"mask {mask.shape}")
    print(f"mask sum of ones {mask.sum()}")
    print(f"mask sum of zeros {(~mask).sum()}")

    unmasked_patches = patches[~mask]
    masked_patches = patches[mask]

    weights_file = "output/cae_base_800e/cae_base_800e_checkpoint-0.pth"

    # Forward
    device = "cuda"
    img = img.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    
    class Args:
        def __init__(self):
            self.decoder_num_classes = 0
            self.decoder_embed_dim=768
            self.regressor_depth=4
            self.decoder_num_heads=12
            self.decoder_layer_scale_init_value=0
            self.decoder_depth=4
            self.fix_init_weight=False
            self.base_momentum=0
    
    model =  model = create_model(
        "cae_base_patch16_224_8k_vocab",
        pretrained=False,
        drop_path_rate=0,
        drop_block_rate=None,
        use_abs_pos_emb=False,
        init_values=0,
        in_chans=13,
        args=Args(),
    )
    model = model.to(device)

    pred_masked_patches, _, _ = model(img, mask)
    
    print(f"pred_masked_patches {pred_masked_patches.shape}")

    pred_masked_patches = pred_masked_patches.to("cpu")
    mask = mask.to("cpu")
    pred_img = torch.full((1, 196, 16**2 * 13), fill_value=float("nan"))
    pred_img[mask] = pred_masked_patches  # add predicted masked patches
    print(f"pred img patchified {pred_img.shape}")
    pred_img = unpatchify(pred_img, c=13).squeeze()
    print(f"pred img patchified {pred_img.shape}")

    pred_out_tiff = "/home/ubuntu/satellite-cae/CAE/output/sample_tiffs/cae_imputed_image.tiff"
    with rasterio.open(pred_out_tiff, 'w', **tiff_meta) as dst:
        dst.write(pred_img.detach().numpy())

    mask_img = torch.full((1, 196, 16**2 * 13), fill_value=float("nan"))
    mask_img[mask] = torch.full((1, 98, mask_img.shape[2]), 1.0)
    mask_img = unpatchify(mask_img, c=13).squeeze()
    save_tiff("mask", mask_img, tiff_meta)
    
    unmasked_img = torch.full((1, 196, 16**2 * 13), fill_value=float("nan"))
    unmasked_img[mask.reshape(1, -1) == 1] = torch.full((1, 98, 16**2 * 13), fill_value=1)
    unmasked_img = unpatchify(unmasked_img, c=13)
    unmasked_img = unmasked_img.squeeze()
    unmasked_out_tiff = "/home/ubuntu/satellite-cae/CAE/output/sample_tiffs/masked.tiff"
    with rasterio.open(unmasked_out_tiff, 'w', **tiff_meta) as dst:
        dst.write(unmasked_img.detach().numpy())



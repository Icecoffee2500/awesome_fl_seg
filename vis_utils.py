import torch
import numpy as np
import matplotlib.pyplot as plt

# 기본 ImageNet mean/std (만약 이미 정규화하지 않았다면 mean/std=None 으로 지정)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# IMAGENET_MEAN = [123.675, 116.28, 103.53]
# IMAGENET_STD  = [58.395, 57.12, 57.375]

def tensor_to_image(img_tensor: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """ image tensor가 한장씩 들어와서 numpy 이미지로 변환됨.

    img_tensor: torch tensor shape (3, H, W) or (H, W, 3) or (B, 3, H, W)
    returns float32 numpy image in range [0, 1], shape (H, W, 3)

    """

    if isinstance(img_tensor, torch.Tensor):
        x = img_tensor.detach().cpu()
    else:
        x = torch.tensor(img_tensor)
    
    # if batch, take first
    if x.ndim == 4:
        x = x[0]
    
    # ensure (3, H, W) = 뭐가 들어와도 (3, H, W)로 만들어주기.
    if x.ndim == 3 and x.shape[0] == 3:
        pass
    elif x.ndim == 3 and x.shape[-1] == 3: # (H, W, 3) -> (3, H, W)
        x = x.permute(2, 0, 1)
    else:
        raise ValueError(f"Unsupported image tensor shape: {x.shape}")
    
    x = x.clone().float() # unnormalize하기 전에는 값들이 음수도 있음.
    if mean is not None and std is not None: # 만약 mean/std가 주어졌다면 정규화 해제
        mean_t = torch.tensor(mean, device=x.device, dtype=x.dtype).view(-1,1,1)
        std_t  = torch.tensor(std,  device=x.device, dtype=x.dtype).view(-1,1,1)
        x = x * std_t + mean_t  # unnormalize # 이제 값들이 0 ~ 1 사이로 변함.
    
    # clamp and convert to HWC [0, 1]
    x = torch.clamp(input=x, min=0, max=1) # clamp는 lower bound, upper bound를 지정해줌. (min, max)
    img = x.permute(1, 2, 0).numpy()
    
    return img

def masks_to_label_map(masks: torch.Tensor):
    """
    Accepts masks that are:
      - (B, 1, H, W) or (B, H, W) -> returns (B, H, W) int labels
      - (B, C, H, W) one-hot/logit -> argmax -> (B, H, W)
    """
    masks = masks.detach().cpu()

    if masks.ndim == 4 and masks.shape[1] == 1: # (B, 1, H, W)
        label = masks[:, 0].long()
    elif masks.ndim == 3: # (B, H, W)
        label = masks.long()
    elif masks.ndim == 4 and masks.shape[1] > 1: # (B, C, H, W)
        label = masks.argmax(dim=1).long() # one-hot or logits -> argmax over channel (C가 만약 class 개수이면 그 중에 값이 제일 높은 index를 선택)
    else:
        raise ValueError(f"Unsupported mask shape: {masks.shape}")

    return label  # (B, H, W) int

# Cityscapes 색상 팔레트 (index 0..19)
def get_cityscapes_palette():
    """
    Returns numpy array shape (20,3) uint8 for Cityscapes-like palette.
    Index meaning (common): 
      0: unlabeled, 1: road, 2: sidewalk, 3: building, 4: wall, 5: fence,
      6: pole, 7: traffic light, 8: traffic sign, 9: vegetation, 10: terrain,
      11: sky, 12: person, 13: rider, 14: car, 15: truck, 16: bus, 17: train,
      18: motorcycle, 19: bicycle
    """
    palette = np.array([
        [29, 185, 84],  # 0 road
        [244,  35, 232],  # 1 sidewalk
        [ 70,  70,  70],  # 2 building
        [102, 102, 156],  # 3 wall
        [190, 153, 153],  # 4 fence
        [153, 153, 153],  # 5 pole
        [250, 170,  30],  # 6 traffic light
        [220, 220,   0],  # 7 traffic sign
        [107, 142,  35],  # 8 vegetation
        [152, 251, 152],  #9 terrain
        [ 70, 130, 180],  #10 sky
        [220,  20,  60],  #11 person
        [255,   0,   0],  #12 rider
        [  0,   0, 142],  #13 car
        [  0,   0,  70],  #14 truck
        [  0,  60, 100],  #15 bus
        [  0,  80, 100],  #16 train
        [  0,   0, 230],  #17 motorcycle
        [119,  11,  32],  #18 bicycle
    ], dtype=np.uint8)
    print(f"palette shape: {palette.shape}")

    return palette

# (예) 기존 visualize_batch에서 palette 만들던 부분을 아래처럼 바꿔 사용하세요:
# palette = get_cityscapes_palette()
# seg_rgb = decode_segmap(label, palette)  # decode_segmap expects palette as uint8 Nx3


def decode_segmap(label_map: torch.Tensor, palette: np.ndarray):
    """
    Args:
        label_map: torch.Tensor(H, W) integer labels
        palette: np.ndarray(num_classes, 3) uint8

    Returns:
        float image [0, 1] (H x W x 3) => color를 입힌 mask image
    """
    height, width = label_map.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    max_id = palette.shape[0] # class 개수

    # safe: if label >= palette length, wrap or clip
    safe_label = np.clip(label_map, 0, max_id - 1)
    rgb = palette[safe_label] # 각 pixel들이 해당하는 class의 색상 코드로 변환됨. (이미지 크기는 그대로)

    return (rgb.astype(np.float32) / 255.0)
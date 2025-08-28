from pathlib import Path
from PIL import Image
import torch
import numpy as np
import random
import os
from torchvision import transforms

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    print(f"🌱 Setting random seed to {seed} for reproducibility")
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # PyTorch deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Set torch number of threads to 1 for deterministic behavior
    torch.set_num_threads(1)
    
    print("✅ Random seed fixed for reproducible results")

def save_data(input, mask, root_dir: Path, save_dir: Path = "visualization"):
    save_path = root_dir / save_dir
    save_path.mkdir(parents=True, exist_ok=True)
    # 만약 img가 Tensor이면 PIL Image로 변환
    if isinstance(input, torch.Tensor):
        # input = transforms.ToPILImage()(input)
        input = transforms.ToPILImage()(input[0].cpu())

    # mask가 Tensor이면 numpy 배열로 변환
    if isinstance(mask, torch.Tensor):
        if mask.ndim == 4 and mask.shape[1] > 1:
            # num_classes 채널인 경우 (logits)
            mask = torch.argmax(mask, dim=1, keepdim=True)
        mask_np = mask[0, 0].detach().cpu().numpy()
        print(f"mask_np shape: {mask_np.shape}")
        print("mask unique values:", np.unique(mask_np))

    else:
        mask_np = np.array(mask)

    # mask를 시각화용 컬러로 변환 (간단히 class id를 grayscale로 표시)
    mask_img = Image.fromarray(mask_np.astype(np.uint8))

    print(f"mask_img shape: {mask_img.size}")

    # 이미지 저장
    input.save(save_path / "img_0.png")
    mask_img.save(save_path / "mask_0.png")

    print(f"Saved image and mask to {save_path}")

def print_keys(keys):
    for k in keys:
        parts = k.split(".")
        indent = "  " * (len(parts) - 1)
        print(f"{indent}{parts[-1]}")
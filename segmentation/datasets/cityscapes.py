import json
from pathlib import Path
import re
from collections import namedtuple
from typing import Optional, Sequence, Union
import cv2
from cv2 import (IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_IGNORE_ORIENTATION,
                 IMREAD_UNCHANGED)
from omegaconf import ListConfig, DictConfig, OmegaConf
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
from .transforms.pipelines import Compose

imread_flags = {
    'color': IMREAD_COLOR,
    'grayscale': IMREAD_GRAYSCALE,
    'unchanged': IMREAD_UNCHANGED,
    'color_ignore_orientation': IMREAD_IGNORE_ORIENTATION | IMREAD_COLOR,
    'grayscale_ignore_orientation':
    IMREAD_IGNORE_ORIENTATION | IMREAD_GRAYSCALE
}

class CityscapesDataset(Dataset):
    """Cityscapes <http://www.cityscapes-dataset.com/> Dataset.
    
    Parameters:
        - root (string): Root directory of dataset where directory 'leftImg8bit' and 'gtFine' or 'gtCoarse' are located.
        - split (string, optional): The image split to use, 'train', 'test' or 'val' if mode="gtFine" otherwise 'train', 'train_extra' or 'val'
        - mode (string, optional): The quality mode to use, 'gtFine' or 'gtCoarse' or 'color'. Can also be a list to output a tuple with all specified target types.
        - transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
        - target_transform (callable, optional): A function/transform that takes in the target and transforms it.
    """

    def __init__(
        self,
        root: str,
        split: str = 'train',
        mode: str = 'gtFine',
        target_type: str = 'semantic',
        pipeline_cfg: Union[list[dict], ListConfig[dict]] = None,
    ):
        # ListConfig로 받아서 여기서 transform의 type을 list[dict]로 맞춰준다.
        if isinstance(pipeline_cfg, ListConfig):
            pipeline_cfg = OmegaConf.to_container(pipeline_cfg, resolve=True)
        elif isinstance(pipeline_cfg, list):
            pass
        else:
            raise TypeError(f'transform must be a list of dict, or ListConfig of dict, but got {type(pipeline_cfg)}')

        self.root = Path(root).expanduser()
        self.mode = mode
        self.target_type = target_type
        self.inputs_dir = self.root / 'leftImg8bit' / split
        self.targets_dir = self.root / self.mode / split
        self.transform = Compose(pipeline_cfg)

        self.split = split
        self.image_file_paths = []
        self.target_file_paths = []
        self.image_metas = []
        self.bboxes = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not self.inputs_dir.is_dir() or not self.targets_dir.is_dir():
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')
        
        for city_path in self.inputs_dir.iterdir():
            city = city_path.name
            img_dir = self.inputs_dir / city
            target_dir = self.targets_dir / city

            for file_path in img_dir.iterdir():
                self.image_file_paths.append(file_path)
                _img_meta = dict(
                    city=city,
                    file_name=file_path.name,
                    bboxes=None
                )
                self.image_metas.append(_img_meta)
                pure_img_name = re.sub(r'_leftImg8bit$', '', file_path.stem) # 순수한 이미지 이름만 # 예를 들면 # bochum_000000_014803
                target_name = f"{pure_img_name}_{self._get_target_suffix(self.mode, self.target_type)}"
                self.target_file_paths.append(target_dir / target_name)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """
        image = Image.open(self.image_file_paths[index]).convert('RGB') # 여기서만 사용됨. (Image.open, Image.convert)
        target = Image.open(self.target_file_paths[index]) # 여기서만 사용됨. (Image.open)

        image = np.array(image) # (1024, 2048, 3)
        target = np.array(target) # (1024, 2048)
        results = dict(img=image, gt_seg_map=target)

        if self.transform:
            results = self.transform(results)
        
        input = results['input']
        target = results['target']
        
        # 여기서 bbox scale 정보 추가.
        # print(f"image.shape: {input.shape}")
        # print(f"target.shape: {target.shape}")
        bboxes_dict = instanceid_map_to_bboxes(target[0])

        # meta 복사 — 절대 self.image_metas를 워커에서 수정하지 않음
        meta = dict(self.image_metas[index])

        # convert dict -> list of [id, x1, y1, x2, y2] (일관된 구조)
        bboxes_list = [[inst_id] + bbox for inst_id, bbox in bboxes_dict.items()]
        meta['bboxes'] = bboxes_list
        # if self.image_metas[index]['bboxes'] is None:
        #     self.image_metas[index]['bboxes'] = bboxes
        
        # if self.image_metas[index]['bboxes'] == {}:
        #     self.image_metas[index]['bboxes'] = bboxes

        # print(f"bboxes: {bboxes}")

        # return input, target, self.image_metas[index]
        return input, target, meta

    def __len__(self):
        return len(self.image_file_paths)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return f'{mode}_instanceIds.png'
        elif target_type == 'semantic':
            return f'{mode}_labelIds.png'
        elif target_type == 'color':
            return f'{mode}_color.png'
        elif target_type == 'polygon':
            return f'{mode}_polygons.json'
        elif target_type == 'depth':
            return f'{mode}_disparity.png'

def instanceid_map_to_bboxes(instance_map, ignore_ids=None, return_xywh=False):
    """
    instance_map: (H,W) integer array where different positive integers denote different instances.
    ignore_ids: set/list of ids to skip (e.g., 0 for background, 255 for void)
    return_xywh: if True, return [x, y, w, h], else return [x1, y1, x2, y2]
    """
    if ignore_ids is None:
        # ignore_ids = {0, 255}  # Cityscapes often uses 0/255 for void/background; 필요에 맞게 조정
        ignore_ids = {255}  # Cityscapes often uses 0/255 for void/background; 필요에 맞게 조정
    ids = np.unique(instance_map)
    # print(f"ids: {ids}")
    bboxes = {}
    for id_ in ids:
        if int(id_) in ignore_ids:
            continue
        mask = (instance_map == id_)
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        if return_xywh:
            b = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]  # width/height 포함
        else:
            b = [x_min, y_min, x_max, y_max]
        bboxes[int(id_)] = b
    return bboxes
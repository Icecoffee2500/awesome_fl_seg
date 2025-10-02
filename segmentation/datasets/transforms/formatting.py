# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from .base_transform import BaseTransform
import numpy as np
import torch
from typing import Union, Sequence
from collections import namedtuple
import torch.nn.functional as F

class PackSegInputs(BaseTransform):
    """Pack the inputs data for the semantic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_path``: filename of the image

        - ``ori_shape``: original shape of the image as a tuple (h, w, c)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w, c).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``pad_shape``: shape of padded images

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be packed from
            ``SegDataSample`` and collected in ``data[img_metas]``.
            Default: ``('img_path', 'ori_shape',
            'img_shape', 'pad_shape', 'scale_factor', 'flip',
            'flip_direction')``
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple(
        "CityscapesClass",
        ["name", "id", "train_id", "category", "category_id", "has_instances", "ignore_in_eval", "color"],
    )

    classes = [
        CityscapesClass("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
        CityscapesClass("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
        CityscapesClass("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
        CityscapesClass("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
        CityscapesClass("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
        CityscapesClass("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
        CityscapesClass("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
        CityscapesClass("wall", 12, 3, "construction", 2, False, False, (102, 102, 156)),
        CityscapesClass("fence", 13, 4, "construction", 2, False, False, (190, 153, 153)),
        CityscapesClass("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)),
        CityscapesClass("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
        CityscapesClass("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)),
        CityscapesClass("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
        CityscapesClass("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)),
        CityscapesClass("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)),
        CityscapesClass("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)),
        CityscapesClass("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
        CityscapesClass("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
        CityscapesClass("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
        CityscapesClass("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
        CityscapesClass("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
        CityscapesClass("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
        CityscapesClass("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
        CityscapesClass("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
        CityscapesClass("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
        CityscapesClass("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
        CityscapesClass("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
        CityscapesClass("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
        CityscapesClass("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
        CityscapesClass("license plate", -1, -1, "vehicle", 7, False, True, (0, 0, 142)),
    ]
    classes_only_car = [
        CityscapesClass("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
        CityscapesClass("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
        CityscapesClass("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
        CityscapesClass("road", 7, 255, "flat", 1, False, False, (128, 64, 128)),
        CityscapesClass("sidewalk", 8, 255, "flat", 1, False, False, (244, 35, 232)),
        CityscapesClass("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
        CityscapesClass("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
        CityscapesClass("building", 11, 255, "construction", 2, False, False, (70, 70, 70)),
        CityscapesClass("wall", 12, 255, "construction", 2, False, False, (102, 102, 156)),
        CityscapesClass("fence", 13, 255, "construction", 2, False, False, (190, 153, 153)),
        CityscapesClass("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)),
        CityscapesClass("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
        CityscapesClass("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)),
        CityscapesClass("pole", 17, 255, "object", 3, False, False, (153, 153, 153)),
        CityscapesClass("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)),
        CityscapesClass("traffic light", 19, 255, "object", 3, False, False, (250, 170, 30)),
        CityscapesClass("traffic sign", 20, 255, "object", 3, False, False, (220, 220, 0)),
        CityscapesClass("vegetation", 21, 255, "nature", 4, False, False, (107, 142, 35)),
        CityscapesClass("terrain", 22, 255, "nature", 4, False, False, (152, 251, 152)),
        CityscapesClass("sky", 23, 255, "sky", 5, False, False, (70, 130, 180)),
        CityscapesClass("person", 24, 255, "human", 6, True, False, (220, 20, 60)),
        CityscapesClass("rider", 25, 255, "human", 6, True, False, (255, 0, 0)),
        CityscapesClass("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)), # 얘만 살려둠.
        CityscapesClass("truck", 27, 255, "vehicle", 7, True, False, (0, 0, 70)),
        CityscapesClass("bus", 28, 255, "vehicle", 7, True, False, (0, 60, 100)),
        CityscapesClass("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
        CityscapesClass("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
        CityscapesClass("train", 31, 255, "vehicle", 7, True, False, (0, 80, 100)),
        CityscapesClass("motorcycle", 32, 255, "vehicle", 7, True, False, (0, 0, 230)),
        CityscapesClass("bicycle", 33, 255, "vehicle", 7, True, False, (119, 11, 32)),
        CityscapesClass("license plate", -1, -1, "vehicle", 7, False, True, (0, 0, 142)),
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)

    # Create a full mapping table from original ID to train_id.
    # This handles all possible pixel values from 0-255.
    # Any ID not in Cityscapes.classes will remain mapped to 255 (ignore_index).
    _id_to_train_id_map = np.full(256, 255, dtype=np.uint8)
    # for c in classes_only_car:
    for c in classes:
        _id_to_train_id_map[c.id] = c.train_id # c.id: 0~33, c.train_id: 0~18 # 여기서 변환 map을 만듦.
    id_to_train_id = _id_to_train_id_map
    # print(f"id_to_train_id: {id_to_train_id}")

    def __init__(self,
                 meta_keys=('img_path', 'seg_map_path', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'reduce_zero_label')):
        self.meta_keys = meta_keys
    
    @classmethod
    def encode_target(cls, target):
        return cls.id_to_train_id[np.array(target)]

    @classmethod
    def decode_target(cls, target):
        target[target == 255] = 19
        #target = target.astype('uint8') + 1
        return cls.train_id_to_color[target]
    
    def _cal_pad_size(self, img_h, img_w, dst_h, dst_w):
        pad_h = dst_h - img_h
        pad_w = dst_w - img_w
        padding_size = (0, pad_w, 0, pad_h)

        return padding_size
    
    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'input' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`SegDataSample`): The annotation info of the
                sample.
        """

        padding_size = None
        if 'crop_size' in results:
            img_pad_val = 0
            seg_pad_val = 255
            
            img_h, img_w = results['img'].shape[:2]
            dst_h, dst_w = results['crop_size']
            padding_size = self._cal_pad_size(img_h, img_w, dst_h, dst_w)
            if padding_size == (0, 0, 0, 0):
                padding_size = None

        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            
            if padding_size is not None:
                # print(f"패딩 들어갑니다잉~!")
                img = F.pad(img, padding_size, value=img_pad_val)

            results['input'] = img
        
        if 'gt_seg_map' in results:
            if len(results['gt_seg_map'].shape) == 2:
                target = results['gt_seg_map'][None, ...].astype(np.int64)
                target = self.encode_target(target)
                target = to_tensor(target)
            else:
                warnings.warn('Please pay attention your ground truth '
                              'segmentation map, usually the segmentation '
                              'map is 2D, but got '
                              f'{results["gt_seg_map"].shape}')
                target = results['gt_seg_map'].astype(np.int64)
                target = self.encode_target(target)
                target = to_tensor(target)
            
            if padding_size is not None:
                # print(f"패딩 들어갑니다잉~!")
                target = F.pad(target, padding_size, value=seg_pad_val)
            
            target = target.long()
            results['target'] = target

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str

class Normalize(BaseTransform):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, results: dict) -> dict:
        x = results['input']
        # ensure float
        if not torch.is_floating_point(x):
            x = x.float()

        # convert mean/std (lists) to tensors on the same device/dtype as x
        mean = torch.tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.tensor(self.std, dtype=x.dtype, device=x.device)

        # reshape mean/std to be broadcastable to x
        if x.ndim == 3:        # (C, H, W)
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
        elif x.ndim == 4:      # (N, C, H, W)
            mean = mean.view(1, -1, 1, 1)
            std = std.view(1, -1, 1, 1)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        results['input'] = (x - mean) / std
        return results


def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int, float]
) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.

    Returns:
        torch.Tensor: the converted data.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')
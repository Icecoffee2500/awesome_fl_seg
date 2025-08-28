# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Callable
from .formatting import PackSegInputs, Normalize
from .basic_augmentation import Resize, RandomResize, RandomCrop, RandomFlip
from .photometric_distortion import PhotoMetricDistortion
# deeplab은 init할 때 이미 callable한 list를 받는데 비해, mmengine 방식은 일단 dict (config)로 받아서 __init__에서 변환해서 사용한다.
class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict], optional): Sequence of transform config dict to be composed.
    """

    def __init__(self, transforms: Optional[Sequence[dict]]):
        self.transforms: list[Callable] = []

        if transforms is None:
            transforms = []

        for transform in transforms:
            if isinstance(transform, dict):
                # print(f"[transform]\n{yaml.dump(transform)}\n")
                transform = build_transform_from_cfg(transform)
                if not callable(transform):
                    raise TypeError(f'transform should be a callable object, '
                                    f'but got {type(transform)}')
                self.transforms.append(transform)
            else:
                raise TypeError(
                    f'transform must be a config dict, '
                    f'but got {type(transform)}')

    def __call__(self, data: dict) -> Optional[dict]:
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        for transform in self.transforms:
            data = transform(data)
            # The transform will return None when it failed to load images or
            # cannot find suitable augmentation parameters to augment the data.
            # Here we simply return None if the transform returns None and the
            # dataset will handle it by randomly selecting another data sample.
            if data is None:
                return None
        return data

    def __repr__(self):
        """Print ``self.transforms`` in sequence.

        Returns:
            str: Formatted string.
        """
        format_string = self.__class__.__name__ + '('
        for transform in self.transforms:
            format_string += '\n'
            format_string += f'    {transform}'
        format_string += '\n)'
        return format_string



def build_transform_from_cfg(_transform_cfg: dict):
    """Build transform from config.
    
    _transform_cfg Example:
        # 하나씩 들어옴.
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
        dict(type='LoadAnnotations'),
        dict(type='PackSegInputs')
    """
    transform_cfg = _transform_cfg.copy()
    transform_type = transform_cfg.pop('type')

    if transform_type == 'Resize':
        transform = Resize(**transform_cfg)
    elif transform_type == 'PackSegInputs':
        transform = PackSegInputs(**transform_cfg)
    elif transform_type == 'Normalize':
        transform = Normalize(**transform_cfg)
    elif transform_type == 'RandomResize':
        transform = RandomResize(**transform_cfg)
    elif transform_type == 'RandomCrop':
        transform = RandomCrop(**transform_cfg)
    elif transform_type == 'RandomFlip':
        transform = RandomFlip(**transform_cfg)
    elif transform_type == 'PhotoMetricDistortion':
        transform = PhotoMetricDistortion(**transform_cfg)
    else:
        raise ValueError(f'Invalid transform type: {transform_type}')

    return transform
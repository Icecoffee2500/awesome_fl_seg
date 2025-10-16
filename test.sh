#!/usr/bin/env bash

# # Centralized Learning
# uv run -m test_cl 'test_checkpoint_path={checkpoint 폴더}/best_model.pth' 'test_data_root={테스트할 해상도의 데이터셋 위치}' 'dataset.test_scale={테스트할 데이터셋의 해상도}'

# # Federated Learning
# uv run -m test_cl 'test_checkpoint_path={checkpoint 폴더}/best_model_fl.pth' 'test_data_root={테스트할 해상도의 데이터셋 위치}' 'dataset.test_scale={테스트할 데이터셋의 해상도}'


# Centralized Learning
uv run -m test_cl 'test_checkpoint_path=performance/0930_00:02/256_256_0/best_model.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]'
uv run -m test_cl 'test_checkpoint_path=performance/0930_00:02/256_256_0/best_model.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]'
uv run -m test_cl 'test_checkpoint_path=performance/0930_00:02/256_256_0/best_model.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]'
uv run -m test_cl 'test_checkpoint_path=performance/0930_00:02/256_256_0/best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
uv run -m test_cl 'test_checkpoint_path=performance/0930_00:02/256_256_0/best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
uv run -m test_cl 'test_checkpoint_path=performance/0930_00:02/256_256_0/best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
uv run -m test_cl 'test_checkpoint_path=performance/0930_00:02/256_256_0/best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'

# Federated Learning
uv run -m test_cl 'test_checkpoint_path=performance/1013_13:21/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]'
uv run -m test_cl 'test_checkpoint_path=performance/1013_13:21/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]'
uv run -m test_cl 'test_checkpoint_path=performance/1013_13:21/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]'
uv run -m test_cl 'test_checkpoint_path=performance/1013_13:21/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
uv run -m test_cl 'test_checkpoint_path=performance/1013_13:21/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
uv run -m test_cl 'test_checkpoint_path=performance/1013_13:21/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
uv run -m test_cl 'test_checkpoint_path=performance/1013_13:21/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'
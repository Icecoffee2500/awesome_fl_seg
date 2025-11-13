#!/usr/bin/env bash

# # Centralized Learning
# uv run -m test_cl 'test_checkpoint_path={checkpoint 폴더}/best_model.pth' 'test_data_root={테스트할 해상도의 데이터셋 위치}' 'dataset.test_scale={테스트할 데이터셋의 해상도}'

# # Federated Learning
# uv run -m test_cl 'test_checkpoint_path={checkpoint 폴더}/best_model_fl.pth' 'test_data_root={테스트할 해상도의 데이터셋 위치}' 'dataset.test_scale={테스트할 데이터셋의 해상도}'

# 1024/768/512 alpha=0.1
uv run -m test_cl 'test_checkpoint_path=performance/1030-2114/fl_1024x1024_768x768_512x512_gpu7_best_model.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1030-2114/fl_1024x1024_768x768_512x512_gpu7_best_model.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1030-2114/fl_1024x1024_768x768_512x512_gpu7_best_model.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1030-2114/fl_1024x1024_768x768_512x512_gpu7_best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1030-2114/fl_1024x1024_768x768_512x512_gpu7_best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1030-2114/fl_1024x1024_768x768_512x512_gpu7_best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1030-2114/fl_1024x1024_768x768_512x512_gpu7_best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]' device_id=6

# 512/384/256 alpha=0.01
uv run -m test_cl 'test_checkpoint_path=performance/1106-1621/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1106-1621/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1106-1621/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1106-1621/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1106-1621/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1106-1621/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1106-1621/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]' device_id=6

# 512/384/256 alpha=0.05
uv run -m test_cl 'test_checkpoint_path=performance/1107-0139/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1107-0139/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1107-0139/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1107-0139/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1107-0139/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1107-0139/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1107-0139/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]' device_id=6

# 512/384/256 alpha=0.3
uv run -m test_cl 'test_checkpoint_path=performance/1107-1041/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1107-1041/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1107-1041/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1107-1041/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1107-1041/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1107-1041/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1107-1041/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]' device_id=6

# 512/384/256 alpha=0.5
uv run -m test_cl 'test_checkpoint_path=performance/1107-1943/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1107-1943/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1107-1943/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1107-1943/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1107-1943/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1107-1943/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1107-1943/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]' device_id=6

# 512/384/256 alpha=0.7
uv run -m test_cl 'test_checkpoint_path=performance/1108-0450/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1108-0450/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1108-0450/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1108-0450/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1108-0450/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1108-0450/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1108-0450/fl_512x512_384x384_256x256_gpu7_best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]' device_id=6
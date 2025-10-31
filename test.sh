#!/usr/bin/env bash

# # Centralized Learning
# uv run -m test_cl 'test_checkpoint_path={checkpoint 폴더}/best_model.pth' 'test_data_root={테스트할 해상도의 데이터셋 위치}' 'dataset.test_scale={테스트할 데이터셋의 해상도}'

# # Federated Learning
# uv run -m test_cl 'test_checkpoint_path={checkpoint 폴더}/best_model_fl.pth' 'test_data_root={테스트할 해상도의 데이터셋 위치}' 'dataset.test_scale={테스트할 데이터셋의 해상도}'

uv run -m test_cl 'test_checkpoint_path=performance/1030-2111/fl_512x512_384x384_256x256_gpu6_best_model.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1030-2111/fl_512x512_384x384_256x256_gpu6_best_model.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1030-2111/fl_512x512_384x384_256x256_gpu6_best_model.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1030-2111/fl_512x512_384x384_256x256_gpu6_best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1030-2111/fl_512x512_384x384_256x256_gpu6_best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1030-2111/fl_512x512_384x384_256x256_gpu6_best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]' device_id=6
uv run -m test_cl 'test_checkpoint_path=performance/1030-2111/fl_512x512_384x384_256x256_gpu6_best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]' device_id=6

# # MRKD Federated Learning
# uv run -m test_cl 'test_checkpoint_path=performance/1029-1439/fl_512x512_384x384_256x256_gpu6_best_model.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]' device_id=6
# uv run -m test_cl 'test_checkpoint_path=performance/1029-1439/fl_512x512_384x384_256x256_gpu6_best_model.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]' device_id=6
# uv run -m test_cl 'test_checkpoint_path=performance/1029-1439/fl_512x512_384x384_256x256_gpu6_best_model.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]' device_id=6
# uv run -m test_cl 'test_checkpoint_path=performance/1029-1439/fl_512x512_384x384_256x256_gpu6_best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]' device_id=6
# uv run -m test_cl 'test_checkpoint_path=performance/1029-1439/fl_512x512_384x384_256x256_gpu6_best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]' device_id=6
# uv run -m test_cl 'test_checkpoint_path=performance/1029-1439/fl_512x512_384x384_256x256_gpu6_best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]' device_id=6
# uv run -m test_cl 'test_checkpoint_path=performance/1029-1439/fl_512x512_384x384_256x256_gpu6_best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]' device_id=6

# uv run -m test_cl 'test_checkpoint_path=performance/1029-1441/fl_1024x1024_768x768_512x512_gpu7_best_model.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]' device_id=6
# uv run -m test_cl 'test_checkpoint_path=performance/1029-1441/fl_1024x1024_768x768_512x512_gpu7_best_model.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]' device_id=6
# uv run -m test_cl 'test_checkpoint_path=performance/1029-1441/fl_1024x1024_768x768_512x512_gpu7_best_model.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]' device_id=6
# uv run -m test_cl 'test_checkpoint_path=performance/1029-1441/fl_1024x1024_768x768_512x512_gpu7_best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]' device_id=6
# uv run -m test_cl 'test_checkpoint_path=performance/1029-1441/fl_1024x1024_768x768_512x512_gpu7_best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]' device_id=6
# uv run -m test_cl 'test_checkpoint_path=performance/1029-1441/fl_1024x1024_768x768_512x512_gpu7_best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]' device_id=6
# uv run -m test_cl 'test_checkpoint_path=performance/1029-1441/fl_1024x1024_768x768_512x512_gpu7_best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]' device_id=6

# # Clustered Federated Learning
# uv run -m test_cl 'test_checkpoint_path=performance/1028-1721/fl_no_RandomResize_no_RandomCropgpu6_best_model.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]' device_id=6
# uv run -m test_cl 'test_checkpoint_path=performance/1028-1721/fl_no_RandomResize_no_RandomCropgpu6_best_model.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]' device_id=6
# uv run -m test_cl 'test_checkpoint_path=performance/1028-1721/fl_no_RandomResize_no_RandomCropgpu6_best_model.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]' device_id=6
# uv run -m test_cl 'test_checkpoint_path=performance/1028-1721/fl_no_RandomResize_no_RandomCropgpu6_best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]' device_id=6
# uv run -m test_cl 'test_checkpoint_path=performance/1028-1721/fl_no_RandomResize_no_RandomCropgpu6_best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]' device_id=6
# uv run -m test_cl 'test_checkpoint_path=performance/1028-1721/fl_no_RandomResize_no_RandomCropgpu6_best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]' device_id=6
# uv run -m test_cl 'test_checkpoint_path=performance/1028-1721/fl_no_RandomResize_no_RandomCropgpu6_best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]' device_id=6

# # Centralized Learning
# uv run -m test_cl 'test_checkpoint_path=performance/1022-1310/fl_no_RandomResize_no_RandomCropgpu0_best_model.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]'
# uv run -m test_cl 'test_checkpoint_path=performance/1022-1310/fl_no_RandomResize_no_RandomCropgpu0_best_model.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]'
# uv run -m test_cl 'test_checkpoint_path=performance/1022-1310/fl_no_RandomResize_no_RandomCropgpu0_best_model.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]'
# uv run -m test_cl 'test_checkpoint_path=performance/1022-1310/fl_no_RandomResize_no_RandomCropgpu0_best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
# uv run -m test_cl 'test_checkpoint_path=performance/1022-1310/fl_no_RandomResize_no_RandomCropgpu0_best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
# uv run -m test_cl 'test_checkpoint_path=performance/1022-1310/fl_no_RandomResize_no_RandomCropgpu0_best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
# uv run -m test_cl 'test_checkpoint_path=performance/1022-1310/fl_no_RandomResize_no_RandomCropgpu0_best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'

# # Federated Learning
# uv run -m test_cl 'test_checkpoint_path=performance/1022-1308/cl_no_RandomResize_no_RandomCrop_best_model.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]'
# uv run -m test_cl 'test_checkpoint_path=performance/1022-1308/cl_no_RandomResize_no_RandomCrop_best_model.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]'
# uv run -m test_cl 'test_checkpoint_path=performance/1022-1308/cl_no_RandomResize_no_RandomCrop_best_model.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]'
# uv run -m test_cl 'test_checkpoint_path=performance/1022-1308/cl_no_RandomResize_no_RandomCrop_best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
# uv run -m test_cl 'test_checkpoint_path=performance/1022-1308/cl_no_RandomResize_no_RandomCrop_best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
# uv run -m test_cl 'test_checkpoint_path=performance/1022-1308/cl_no_RandomResize_no_RandomCrop_best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
# uv run -m test_cl 'test_checkpoint_path=performance/1022-1308/cl_no_RandomResize_no_RandomCrop_best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'


# # Centralized Learning
# # 256x256
# uv run -m test_cl 'test_checkpoint_path=performance/1014_23:16/256_256_1/best_model.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_23:16/256_256_1/best_model.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_23:16/256_256_1/best_model.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_23:16/256_256_1/best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_23:16/256_256_1/best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_23:16/256_256_1/best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_23:16/256_256_1/best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'

# # 320x320
# uv run -m test_cl 'test_checkpoint_path=performance/1016_05:45/320_320_1/best_model.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]'
# uv run -m test_cl 'test_checkpoint_path=performance/1016_05:45/320_320_1/best_model.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]'
# uv run -m test_cl 'test_checkpoint_path=performance/1016_05:45/320_320_1/best_model.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]'
# uv run -m test_cl 'test_checkpoint_path=performance/1016_05:45/320_320_1/best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
# uv run -m test_cl 'test_checkpoint_path=performance/1016_05:45/320_320_1/best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
# uv run -m test_cl 'test_checkpoint_path=performance/1016_05:45/320_320_1/best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
# uv run -m test_cl 'test_checkpoint_path=performance/1016_05:45/320_320_1/best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'

# # 384x384
# uv run -m test_cl 'test_checkpoint_path=performance/1015_10:43/384_384_0/best_model.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_10:43/384_384_0/best_model.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_10:43/384_384_0/best_model.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_10:43/384_384_0/best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_10:43/384_384_0/best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_10:43/384_384_0/best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_10:43/384_384_0/best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'

# # 448x448
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:12/448_448_3/best_model.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:12/448_448_3/best_model.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:12/448_448_3/best_model.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:12/448_448_3/best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:12/448_448_3/best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:12/448_448_3/best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:12/448_448_3/best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'

# # 512x512
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:11/512_512_2/best_model.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:11/512_512_2/best_model.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:11/512_512_2/best_model.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:11/512_512_2/best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:11/512_512_2/best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:11/512_512_2/best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:11/512_512_2/best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'

# # 768x768
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:10/768_768_1/best_model.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:10/768_768_1/best_model.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:10/768_768_1/best_model.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:10/768_768_1/best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:10/768_768_1/best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:10/768_768_1/best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:10/768_768_1/best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'

# # 1024x1024
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:09/1024_1024_0/best_model.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:09/1024_1024_0/best_model.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:09/1024_1024_0/best_model.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:09/1024_1024_0/best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:09/1024_1024_0/best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:09/1024_1024_0/best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_12:09/1024_1024_0/best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'



# # --------------------------------------------------------------------------------------------------------------------------------------------
# # Federated Learning

# # 1024 1024 1024
# uv run -m test_cl 'test_checkpoint_path=performance/1014_19:40/fl_2/best_model_fl.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_19:40/fl_2/best_model_fl.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_19:40/fl_2/best_model_fl.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_19:40/fl_2/best_model_fl.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_19:40/fl_2/best_model_fl.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_19:40/fl_2/best_model_fl.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_19:40/fl_2/best_model_fl.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'

# # 768 768 768
# uv run -m test_cl 'test_checkpoint_path=performance/1014_19:31/fl_3/best_model_fl.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_19:31/fl_3/best_model_fl.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_19:31/fl_3/best_model_fl.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_19:31/fl_3/best_model_fl.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_19:31/fl_3/best_model_fl.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_19:31/fl_3/best_model_fl.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
# uv run -m test_cl 'test_checkpoint_path=performance/1014_19:31/fl_3/best_model_fl.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'

# # 512 512 512
# uv run -m test_cl 'test_checkpoint_path=performance/1015_17:57/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_17:57/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_17:57/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_17:57/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_17:57/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_17:57/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_17:57/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'

# # 384 384 384
# uv run -m test_cl 'test_checkpoint_path=performance/1015_06:15/fl_1/best_model_fl.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_06:15/fl_1/best_model_fl.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_06:15/fl_1/best_model_fl.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_06:15/fl_1/best_model_fl.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_06:15/fl_1/best_model_fl.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_06:15/fl_1/best_model_fl.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_06:15/fl_1/best_model_fl.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'

# # 256 256 256
# uv run -m test_cl 'test_checkpoint_path=performance/1015_18:04/fl_2/best_model_fl.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_18:04/fl_2/best_model_fl.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_18:04/fl_2/best_model_fl.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_18:04/fl_2/best_model_fl.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_18:04/fl_2/best_model_fl.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_18:04/fl_2/best_model_fl.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_18:04/fl_2/best_model_fl.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'

# # 512 384 256
# uv run -m test_cl 'test_checkpoint_path=performance/1015_09:49/fl_3/best_model_fl.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_09:49/fl_3/best_model_fl.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_09:49/fl_3/best_model_fl.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_09:49/fl_3/best_model_fl.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_09:49/fl_3/best_model_fl.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_09:49/fl_3/best_model_fl.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
# uv run -m test_cl 'test_checkpoint_path=performance/1015_09:49/fl_3/best_model_fl.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'

# # 512 256 256
# uv run -m test_cl 'test_checkpoint_path=performance/1016_03:43/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes' 'dataset.test_scale=[2048, 1024]'
# uv run -m test_cl 'test_checkpoint_path=performance/1016_03:43/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]'
# uv run -m test_cl 'test_checkpoint_path=performance/1016_03:43/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]'
# uv run -m test_cl 'test_checkpoint_path=performance/1016_03:43/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
# uv run -m test_cl 'test_checkpoint_path=performance/1016_03:43/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
# uv run -m test_cl 'test_checkpoint_path=performance/1016_03:43/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
# uv run -m test_cl 'test_checkpoint_path=performance/1016_03:43/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'
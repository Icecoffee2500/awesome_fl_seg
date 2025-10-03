#!/usr/bin/env bash

# # 256_256_0
# # uv run -m test_cl 'test_checkpoint_path=performance/0930_00:02/256_256_0/best_model.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]'
# uv run -m test_cl 'test_checkpoint_path=performance/0930_00:02/256_256_0/best_model.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]'
# # uv run -m test_cl 'test_checkpoint_path=performance/0930_00:02/256_256_0/best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
# # uv run -m test_cl 'test_checkpoint_path=performance/0930_00:02/256_256_0/best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
# # uv run -m test_cl 'test_checkpoint_path=performance/0930_00:02/256_256_0/best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
# # uv run -m test_cl 'test_checkpoint_path=performance/0930_00:02/256_256_0/best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'

# # 320_320_1
# # uv run -m test_cl 'test_checkpoint_path=performance/0930_00:04/320_320_1/best_model.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]'
# uv run -m test_cl 'test_checkpoint_path=performance/0930_00:04/320_320_1/best_model.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]'
# # uv run -m test_cl 'test_checkpoint_path=performance/0930_00:04/320_320_1/best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
# # uv run -m test_cl 'test_checkpoint_path=performance/0930_00:04/320_320_1/best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
# # uv run -m test_cl 'test_checkpoint_path=performance/0930_00:04/320_320_1/best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
# # uv run -m test_cl 'test_checkpoint_path=performance/0930_00:04/320_320_1/best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'

# 384_384_2
# uv run -m test_cl 'test_checkpoint_path=performance/0930_00:05/384_384_2/best_model.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]'
# uv run -m test_cl 'test_checkpoint_path=performance/0930_00:05/384_384_2/best_model.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]'
uv run -m test_cl 'test_checkpoint_path=performance/0930_00:05/384_384_2/best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
uv run -m test_cl 'test_checkpoint_path=performance/0930_00:05/384_384_2/best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
uv run -m test_cl 'test_checkpoint_path=performance/0930_00:05/384_384_2/best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
uv run -m test_cl 'test_checkpoint_path=performance/0930_00:05/384_384_2/best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'

# 448_448_3
uv run -m test_cl 'test_checkpoint_path=performance/0930_00:07/448_448_3/best_model.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]'
# uv run -m test_cl 'test_checkpoint_path=performance/0930_00:07/448_448_3/best_model.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]'
uv run -m test_cl 'test_checkpoint_path=performance/0930_00:07/448_448_3/best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
uv run -m test_cl 'test_checkpoint_path=performance/0930_00:07/448_448_3/best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
uv run -m test_cl 'test_checkpoint_path=performance/0930_00:07/448_448_3/best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
uv run -m test_cl 'test_checkpoint_path=performance/0930_00:07/448_448_3/best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'

# 512 / 384 / 256 (fl)
uv run -m test_cl 'test_checkpoint_path=performance/0930_10:35/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]'
# uv run -m test_cl 'test_checkpoint_path=performance/0930_10:35/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]'
uv run -m test_cl 'test_checkpoint_path=performance/0930_10:35/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
uv run -m test_cl 'test_checkpoint_path=performance/0930_10:35/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
uv run -m test_cl 'test_checkpoint_path=performance/0930_10:35/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
uv run -m test_cl 'test_checkpoint_path=performance/0930_10:35/fl_0/best_model_fl.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'

# 512 / 256 / 256 (fl)
uv run -m test_cl 'test_checkpoint_path=performance/0930_10:35/fl_1/best_model_fl.pth' 'test_data_root=data/cityscapes_1536x768' 'dataset.test_scale=[1536, 768]'
# uv run -m test_cl 'test_checkpoint_path=performance/0930_10:35/fl_1/best_model_fl.pth' 'test_data_root=data/cityscapes_1024x512' 'dataset.test_scale=[1024, 512]'
uv run -m test_cl 'test_checkpoint_path=performance/0930_10:35/fl_1/best_model_fl.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
uv run -m test_cl 'test_checkpoint_path=performance/0930_10:35/fl_1/best_model_fl.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
uv run -m test_cl 'test_checkpoint_path=performance/0930_10:35/fl_1/best_model_fl.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
uv run -m test_cl 'test_checkpoint_path=performance/0930_10:35/fl_1/best_model_fl.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'

# 512_512_2
uv run -m test_cl 'test_checkpoint_path=performance/0903_16:21/512_512_2/best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
# uv run -m test_cl 'test_checkpoint_path=performance/0903_16:21/512_512_2/best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
uv run -m test_cl 'test_checkpoint_path=performance/0903_16:21/512_512_2/best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
uv run -m test_cl 'test_checkpoint_path=performance/0903_16:21/512_512_2/best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'

# 768_768_1
uv run -m test_cl 'test_checkpoint_path=performance/0903_16:20/768_768_1/best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
# uv run -m test_cl 'test_checkpoint_path=performance/0903_16:20/768_768_1/best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
uv run -m test_cl 'test_checkpoint_path=performance/0903_16:20/768_768_1/best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
uv run -m test_cl 'test_checkpoint_path=performance/0903_16:20/768_768_1/best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'

# 1024_1024_0
uv run -m test_cl 'test_checkpoint_path=performance/0903_16:19/1024_1024_0/best_model.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
# uv run -m test_cl 'test_checkpoint_path=performance/0903_16:19/1024_1024_0/best_model.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
uv run -m test_cl 'test_checkpoint_path=performance/0903_16:19/1024_1024_0/best_model.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
uv run -m test_cl 'test_checkpoint_path=performance/0903_16:19/1024_1024_0/best_model.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'

# 1024 / 768 / 512 (fl)
uv run -m test_cl 'test_checkpoint_path=performance/0903_16:18/fl_3/best_model_fl.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
# uv run -m test_cl 'test_checkpoint_path=performance/0903_16:18/fl_3/best_model_fl.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
uv run -m test_cl 'test_checkpoint_path=performance/0903_16:18/fl_3/best_model_fl.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
uv run -m test_cl 'test_checkpoint_path=performance/0903_16:18/fl_3/best_model_fl.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'

# 1024 / 1024 / 1024 (fl)
uv run -m test_cl 'test_checkpoint_path=performance/0908_11:41/fl_3/best_model_fl.pth' 'test_data_root=data/cityscapes_896x448' 'dataset.test_scale=[896, 448]'
# uv run -m test_cl 'test_checkpoint_path=performance/0908_11:41/fl_3/best_model_fl.pth' 'test_data_root=data/cityscapes_768x384' 'dataset.test_scale=[768, 384]'
uv run -m test_cl 'test_checkpoint_path=performance/0908_11:41/fl_3/best_model_fl.pth' 'test_data_root=data/cityscapes_640x320' 'dataset.test_scale=[640, 320]'
uv run -m test_cl 'test_checkpoint_path=performance/0908_11:41/fl_3/best_model_fl.pth' 'test_data_root=data/cityscapes_512x256' 'dataset.test_scale=[512, 256]'
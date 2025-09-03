uv run -m train_cl 'dataset.crop_size=[1024,1024]' device_id=0
uv run -m train_cl 'dataset.crop_size=[768,768]' device_id=1
uv run -m train_cl 'dataset.crop_size=[512,512]' device_id=2
uv run -m train_fl 'dataset.crop_size=[768,768]' device_id=3
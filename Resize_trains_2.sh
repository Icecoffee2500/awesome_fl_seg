#!/usr/bin/env bash

# CL
uv run -m train_cl 'dataset.crop_size=[512,512]' device_id=2

# FL
uv run -m train_fl -- 'fl.target_resolutions={0:[1024,1024],1:[1024,1024],2:[1024,1024]}' device_id=2
uv run -m train_fl -- 'fl.target_resolutions={0:[256,256],1:[256,256],2:[256,256]}' device_id=2
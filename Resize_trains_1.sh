#!/usr/bin/env bash

# CL
uv run -m train_cl 'dataset.crop_size=[768,768]' device_id=1
uv run -m train_cl 'dataset.crop_size=[256,256]' device_id=1

# FL
# uv run -m train_fl -- 'fl.target_resolutions={0:[1024,1024],1:[1024,1024],2:[1024,1024]}' device_id=0
# uv run -m train_fl -- 'fl.target_resolutions={0:[768,768],1:[768,768],2:[768,768]}' device_id=0
# uv run -m train_fl -- 'fl.target_resolutions={0:[512,512],1:[512,512],2:[512,512]}' device_id=1
uv run -m train_fl -- 'fl.target_resolutions={0:[384,384],1:[384,384],2:[384,384]}' device_id=2
# uv run -m train_fl -- 'fl.target_resolutions={0:[256,256],1:[256,256],2:[256,256]}' device_id=3

# uv run -m train_fl -- 'fl.target_resolutions={0:[512,512],1:[384,384],2:[256,256]}' device_id=1
# uv run -m train_fl -- 'fl.target_resolutions={0:[512,512],1:[256,256],2:[256,256]}' device_id=1
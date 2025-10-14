#!/usr/bin/env bash

# CL
uv run -m train_cl 'dataset.crop_size=[1024,1024]' device_id=0
uv run -m train_cl 'dataset.crop_size=[384,384]' device_id=0

# FL
uv run -m train_fl -- 'fl.target_resolutions={0:[512,512],1:[512,512],2:[512,512]}' device_id=0

uv run -m train_fl -- 'fl.target_resolutions={0:[512,512],1:[256,256],2:[256,256]}' device_id=0
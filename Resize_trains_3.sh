#!/usr/bin/env bash

# CL
uv run -m train_cl 'dataset.crop_size=[448,448]' device_id=2

# FL
uv run -m train_fl -- 'fl.target_resolutions={0:[768,768],1:[768,768],2:[768,768]}' device_id=3

uv run -m train_fl -- 'fl.target_resolutions={0:[512,512],1:[384,384],2:[256,256]}' device_id=3
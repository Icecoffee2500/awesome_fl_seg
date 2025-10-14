#!/usr/bin/env bash

# CL
uv run -m train_cl 'dataset.crop_size=[768,768]' device_id=1
uv run -m train_cl 'dataset.crop_size=[256,256]' device_id=1

# FL
uv run -m train_fl -- 'fl.target_resolutions={0:[384,384],1:[384,384],2:[384,384]}' device_id=1
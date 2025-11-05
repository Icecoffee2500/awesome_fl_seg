#!/usr/bin/env bash

# uv run -m train_fl -- 'fl.target_resolutions={0:[1024,1024],1:[768,768],2:[512,512]}' device_id=7
# uv run -m train_fl -- 'fl.target_resolutions={0:[512,512],1:[512,512],2:[512,512]}' device_id=7
uv run -m train_fl -- 'fl.target_resolutions={0:[384,384],1:[384,384],2:[384,384]}' device_id=7
uv run -m train_fl -- 'fl.target_resolutions={0:[320,320],1:[320,320],2:[320,320]}' device_id=7
uv run -m train_fl -- 'fl.target_resolutions={0:[256,256],1:[256,256],2:[256,256]}' device_id=7
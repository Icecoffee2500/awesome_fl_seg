#!/usr/bin/env bash

# uv run -m train_fl device_id=6
# uv run -m train_fl -- 'fl.target_resolutions={0:[1024,1024],1:[1024,1024],2:[1024,1024]}' device_id=6
uv run -m train_fl -- 'fl.target_resolutions={0:[768,768],1:[768,768],2:[768,768]}' device_id=6
uv run -m train_fl -- 'fl.target_resolutions={0:[448,448],1:[448,448],2:[448,448]}' device_id=6
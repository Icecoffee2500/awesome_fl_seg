#!/usr/bin/env bash

# uv run -m train_fl device_id=6
# uv run -m train_fl -- 'fl.target_resolutions={0:[1024,1024],1:[1024,1024],2:[1024,1024]}' device_id=6
# uv run -m train_fl -- 'fl.target_resolutions={0:[1024,1024],1:[1024,1024],2:[1024,1024]}' device_id=6 fl.mrkd_alpha=0.01
# uv run -m train_fl -- 'fl.target_resolutions={0:[1024,1024],1:[1024,1024],2:[1024,1024]}' device_id=6 fl.mrkd_alpha=0.05
# uv run -m train_fl -- 'fl.target_resolutions={0:[1024,1024],1:[1024,1024],2:[1024,1024]}' device_id=6 fl.mrkd_alpha=0.3

uv run -m train_fl -- 'fl.target_resolutions={0:[1024,1024],1:[768,768],2:[512,512]}' device_id=6 fl.mrkd_alpha=0.01
uv run -m train_fl -- 'fl.target_resolutions={0:[1024,1024],1:[768,768],2:[512,512]}' device_id=6 fl.mrkd_alpha=0.05
uv run -m train_fl -- 'fl.target_resolutions={0:[1024,1024],1:[768,768],2:[512,512]}' device_id=6 fl.mrkd_alpha=0.3
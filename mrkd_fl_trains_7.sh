#!/usr/bin/env bash

uv run -m train_fl -- 'fl.target_resolutions={0:[1024,1024],1:[768,768],2:[512,512]}' device_id=7
uv run -m train_fl -- 'fl.target_resolutions={0:[512,512],1:[512,512],2:[512,512]}' device_id=7
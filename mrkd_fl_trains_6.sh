#!/usr/bin/env bash

uv run -m train_fl device_id=6
uv run -m train_fl -- 'fl.target_resolutions={0:[1024,1024],1:[1024,1024],2:[1024,1024]}' device_id=6
#!/usr/bin/env bash

uv run -m create_mr_cityscapes 512 256
uv run -m create_mr_cityscapes 640 320
uv run -m create_mr_cityscapes 768 384
uv run -m create_mr_cityscapes 896 448

# uv run -m create_mr_cityscapes 1024 512

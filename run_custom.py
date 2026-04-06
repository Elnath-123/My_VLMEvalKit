"""Thin wrapper around run.py: reads model definitions from a JSON file
(via MODEL_CONFIG_JSON env var) and injects them into supported_VLM,
so that --model / --data can work without touching config.py.
"""
import json
import os
from functools import partial

# Import run first — its module-level code handles GPU partitioning
# (setting CUDA_VISIBLE_DEVICES per LOCAL_RANK) before any CUDA init.
from run import main
from vlmeval.smp import load_env

from vlmeval.config import supported_VLM
import vlmeval.vlm

config_path = os.environ.get('MODEL_CONFIG_JSON', '')
if config_path:
    with open(config_path) as f:
        cfg = json.load(f)
    for name, model_cfg in cfg['model'].items():
        cls_name = model_cfg.pop('class')
        cls = getattr(vlmeval.vlm, cls_name)
        supported_VLM[name] = partial(cls, **model_cfg)

if __name__ == '__main__':
    load_env()
    main()

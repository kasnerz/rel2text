#!/usr/bin/env python

import argparse
import logging
import numpy as np
import os
import re
import torch

import pytorch_lightning as pl

from pprint import pprint as pp
from inference import (
    DefaultInferenceModule,
)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default="experiments", type=str,
        help="Base directory of the experiment.")
    parser.add_argument("--experiment", type=str, default=None,
        help="Name of the experiment directory from which the model is loaded if `--model_name` is not specified.")
    parser.add_argument("--model_name", type=str, default=None,
        help="Name of the pretrained model used for prediction if `--experiment` is not specified.")
    parser.add_argument("--seed", default=42, type=int,
        help="Random seed.")
    parser.add_argument("--max_threads", default=8, type=int,
        help="Maximum number of threads.")
    parser.add_argument("--beam_size", default=5, type=int,
        help="Beam size.")
    parser.add_argument("--gpus", default=1, type=int,
        help="Number of GPUs.")
    parser.add_argument("--max_length", type=int, default=1024,
        help="Maximum number of tokens per example")
    parser.add_argument("--checkpoint", type=str, default="model.ckpt",
        help="Override the default checkpoint name 'model.ckpt'.")
    args = parser.parse_args()

    logger.info(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.max_threads)
    
    if args.experiment is not None and args.model_name is not None:
        raise ValueError("The parameters `experiment` and `model_name` are mutually exclusive,\
            please specify only one of these.")

    elif args.experiment is None and args.model_name is None:
        raise ValueError("Please specify one of the following parameters: `experiment` OR `model_name`")

    inference_module_cls = DefaultInferenceModule

    if args.experiment:
        model_path = os.path.join(args.exp_dir, args.experiment, args.checkpoint)
        dm = inference_module_cls(args, model_path=model_path)

    elif args.model_name:
        dm = inference_module_cls(args)

    while True:
        # wait for user input
        s = input("[In]: ")
        out = dm.predict(s)
        
        print("[Out]:")
        pp(out, width=300)
        print("============")


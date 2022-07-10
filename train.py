#!/usr/bin/env python3

import logging
import argparse
import os
import pytorch_lightning as pl

from model import (
    TrainingModule,
    DefaultTrainingModule,
)
from dataloader import (
    DataModule,
)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = TrainingModule.add_model_specific_args(parser)

    parser.add_argument("--model_name", type=str,default="facebook/bart-base",
        help="Name of the model from the Huggingface Transformers library.")
    parser.add_argument("--model_path", type=str, default=None,
        help="Path to the saved checkpoint to be loaded.")
    parser.add_argument("--in_dir", type=str, required=True,
        help="Input directory with the data.")
    parser.add_argument("--batch_size", type=int, default=8,
        help="Batch size for finetuning the model")
    parser.add_argument("--out_dir", type=str, default="experiments",
        help="Output directory")
    parser.add_argument("--checkpoint_name", type=str, default="model",
        help="Name of the checkpoint (default='model')")
    parser.add_argument("--experiment", type=str, required=True,
        help="Name used for the experiment directory")
    parser.add_argument("--max_length", type=int, default=512,
        help="Maximum number of tokens per example")
    parser.add_argument("--seed", default=42, type=int,
        help="Random seed.")
    parser.add_argument("--max_threads", default=8, type=int,
        help="Maximum number of CPU threads.")
    parser.add_argument("--resume_training", action="store_true",
        help="Resume training from the loaded checkpoint (useful if training was interrupted).")
    parser.add_argument("--bleu_ref", type=str, default=None,
        help="Reference file for computing BLEU score. If specified, the validation checkpoint is \
        selected according to BLEU score instead of loss.")
    
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    logger.info("Initializing...")
    logger.info(args)

    pl.seed_everything(args.seed)

    training_module_cls = DefaultTrainingModule
    data_module_cls = DataModule

    data_module = data_module_cls(args, special_tokens=training_module_cls.special_tokens)
    data_module.prepare_data()
    data_module.setup('fit')
    resume_from_checkpoint = None

    if args.model_path:
        model = training_module_cls.load_from_checkpoint(
            args.model_path,
            datamodule=data_module
        )
        if args.resume_training:
            resume_from_checkpoint = args.model_path
    else:
        if args.resume_training:
            raise ValueError("Please specify the path to the trained model (`--model_path`) for resuming training.")

        model = training_module_cls(args, datamodule=data_module)

    ckpt_out_dir = os.path.join(args.out_dir,
        args.experiment
    )
    
    if args.bleu_ref:
        monitor = "bleu-4/val"
        mode = "max"

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_out_dir,
            filename=args.checkpoint_name,
            save_top_k=1,
            verbose=True,
            monitor=monitor,
            mode=mode
        )
    else:
        logger.warning("Not monitoring validation score - use with fewshot experiments only.")
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=ckpt_out_dir,
            filename=args.checkpoint_name,
            save_last=True,
            save_top_k=0,
            verbose=True,
            monitor=None,
        )
        checkpoint_callback.CHECKPOINT_NAME_LAST = "model"

    trainer = pl.Trainer.from_argparse_args(args,
        callbacks=[checkpoint_callback], 
        accelerator="auto",
        strategy='dp',
        resume_from_checkpoint=resume_from_checkpoint
    )
    trainer.fit(model, data_module)

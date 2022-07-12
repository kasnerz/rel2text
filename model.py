#!/usr/bin/env python3

import numpy as np
import os
import logging
import argparse
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import sacrebleu
from torch.utils.data import DataLoader, Dataset

from preprocess import (
    HEAD_MARKER,
    REL_MARKER,
    TAIL_MARKER,
    HEAD_DESC_MARKER,
    REL_DESC_MARKER,
    TAIL_DESC_MARKER,
)

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoConfig,
    AutoTokenizer,
    get_scheduler
)
from torch.optim import (
    AdamW,
    Adagrad
)
logger = logging.getLogger(__name__)


def add_special_tokens(tokenizer, model, tokens):
    special_tokens_dict = {'additional_special_tokens': tokens}
    tokenizer.add_special_tokens(special_tokens_dict)

    # logger.info(f"Adding tokens to model vocabulary: {tokens}")

    if model is not None:
        model.resize_token_embeddings(len(tokenizer))


class TrainingModule(pl.LightningModule):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.save_hyperparameters(ignore=["datamodule", "bleu_ref"])
        self.special_tokens = None
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name,
                                                       use_fast=True)
        self.datamodule = kwargs.get("datamodule", None)
        self.validate_with_bleu = False

        if hasattr(self.args, "bleu_ref") \
            and self.args.bleu_ref is not None\
            and self.args.bleu_ref != "":

            self.decoded = []
            self.validate_with_bleu = True
            self.refs, self.ref_len = self.load_refs(self.args.bleu_ref)

    def load_refs(self, bleu_ref):
        with open(bleu_ref) as f:
            if bleu_ref.endswith("json"):
                j = json.load(f)
                refs = [x["out"] for x in j["data"]]
            else:
                refs = f.read().rstrip("\n").split("\n\n")

            ref = [ref_group.split("\n") for ref_group in refs]
            max_refs = max([len(refs) for refs in ref])
            refs_transposed = [[refs[i] if len(refs) > i else None for refs in ref] 
                                            for i in range(max_refs)]
            return refs_transposed, len(refs)


    def forward(self, **inputs):
        out = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )
        return {"loss": out["loss"], "logits": out["logits"]}


    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs["loss"]

        self.log('loss/train', loss, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs["loss"]

        if self.validate_with_bleu:
            out_batch = self.model.generate(batch["input_ids"], 
                max_length=self.args.max_length,
                num_beams=1,
                num_return_sequences=1)
            sentences = self.tokenizer.batch_decode(out_batch, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            self.decoded += sentences

        self.log('loss/val', loss, prog_bar=True)
        return loss

    def validation_epoch_end(self, outputs):
        if self.validate_with_bleu:
            if len(self.decoded) != self.ref_len:
                # validation sanity check
                logger.warn(f"len(self.decoded)={len(self.decoded)} vs. len(self.ref)={len(self.refs)} (validation sanity check?).")
                self.log('bleu-4/val', torch.tensor(0.))
                self.decoded.clear()
                return

            bleu = torch.tensor(sacrebleu.corpus_bleu(self.decoded, self.refs).score)

            self.log('bleu-4/val', bleu)
            self.decoded.clear()

    def test_step(self, batch, batch_idx):
        out = self.model.generate(batch["input_ids"], 
            max_length=self.args.max_length,
            num_beams=self.beam_size_decode,
            num_return_sequences=1)
        
        out = self.tokenizer.batch_decode(out, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        for idx, o in enumerate(out):
            logger.info(f"[{batch_idx * len(out) + idx}] {o}")
            self.out_file_handle.write(o + "\n")


    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon,
            betas=(self.args.adam_beta1, self.args.adam_beta2)
        )
        total_steps = self.args.max_steps if (self.args.max_steps is not None and self.args.max_steps != -1) else len(
            self.datamodule.train_dataloader()) * self.args.max_epochs
        warmup_steps = total_steps * self.args.warmup_proportion

        scheduler = get_scheduler(
            "polynomial",
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        logger.info(f"Using Adam optimizer")
        logger.info(f"Learning rate: {self.args.learning_rate}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")

        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]
        

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser],
                                         add_help=False)

        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-06, type=float)
        parser.add_argument("--adam_beta1", default=0.9, type=float)
        parser.add_argument("--adam_beta2", default=0.98, type=float)
        parser.add_argument("--warmup_proportion", default=0.1, type=float)
        parser.add_argument("--weight_decay", default=0.01, type=float)
        parser.add_argument("--dropout", default=0.2, type=float)
        parser.add_argument("--label_smoothing", default=0.1, type=float)

        return parser


class Seq2SeqTrainingModule(TrainingModule):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name,
            return_dict=True
        )
        add_special_tokens(self.tokenizer, self.model, tokens=self.__class__.special_tokens)


class DefaultTrainingModule(Seq2SeqTrainingModule):
    special_tokens = [HEAD_MARKER, REL_MARKER, TAIL_MARKER, HEAD_DESC_MARKER, REL_DESC_MARKER, TAIL_DESC_MARKER]

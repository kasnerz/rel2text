#!/usr/bin/env python3

import os
import argparse
import logging
import data
import json
import random
import re
import numpy as np
from collections import defaultdict
from utils.tokenizer import Tokenizer
from data import get_dataset_class_by_name

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

MASK_TOKEN = "<mask>"
HEAD_MARKER = "<head>"
REL_MARKER = "<rel>"
TAIL_MARKER = "<tail>"
HEAD_DESC_MARKER = "<head_desc>"
REL_DESC_MARKER = "<rel_desc>"
TAIL_DESC_MARKER = "<tail_desc>"

class Preprocessor:
    def __init__(self, dataset, mode):
        self.dataset = dataset
        self.mode = mode
        self.tokenizer = Tokenizer()

    def _linearize_triples(self, triples):
        out = []

        for triple in triples:
            out.append(HEAD_MARKER)
            out.append(self.tokenizer.normalize(triple.head, remove_quotes=True, remove_parentheses=True))
            out.append(REL_MARKER)
            out.append(self.tokenizer.normalize(triple.rel, remove_quotes=True, remove_parentheses=True))
            out.append(TAIL_MARKER)
            out.append(self.tokenizer.normalize(triple.tail, remove_quotes=True, remove_parentheses=True))

        out = " ".join(out)

        return out


    def _linearize_triples_mask_rel(self, triples):
        out = []

        for triple in triples:
            out.append(HEAD_MARKER)
            out.append(self.tokenizer.normalize(triple.head, remove_quotes=True, remove_parentheses=True))
            out.append(REL_MARKER)
            out.append(MASK_TOKEN)
            out.append(TAIL_MARKER)
            out.append(self.tokenizer.normalize(triple.tail, remove_quotes=True, remove_parentheses=True))

        out = " ".join(out)

        return out

    def _linearize_triples_desc_cat(self, triples):
        out = []

        for triple in triples:
            out.append(HEAD_MARKER)
            out.append(self.tokenizer.normalize(triple.head, remove_quotes=True, remove_parentheses=True))
            out.append(REL_MARKER)
            out.append(self.tokenizer.normalize(triple.rel, remove_quotes=True, remove_parentheses=True))
            out.append(TAIL_MARKER)
            out.append(self.tokenizer.normalize(triple.tail, remove_quotes=True, remove_parentheses=True))
            out.append(REL_DESC_MARKER)
            out.append(self.tokenizer.normalize(triple.rel_desc, remove_quotes=False, remove_parentheses=False))

        out = " ".join(out)

        return out


    def _linearize_triples_desc_repl(self, triples):
        out = []

        for triple in triples:
            out.append(HEAD_MARKER)
            out.append(self.tokenizer.normalize(triple.head, remove_quotes=True, remove_parentheses=True))
            out.append(REL_DESC_MARKER)
            out.append(self.tokenizer.normalize(triple.rel_desc, remove_quotes=False, remove_parentheses=False))
            out.append(TAIL_MARKER)
            out.append(self.tokenizer.normalize(triple.tail, remove_quotes=True, remove_parentheses=True))

        out = " ".join(out)

        return out

    def create_examples(self, entry):
        """
        Generates training examples from an entry in the dataset
        """
        examples = []

        if self.mode == "full":
            inp = self._linearize_triples(entry.data)

        elif self.mode == "desc_cat":
            inp = self._linearize_triples_desc_cat(entry.data)

        elif self.mode == "desc_repl":
            inp = self._linearize_triples_desc_repl(entry.data)

        elif self.mode == "mask_rel":
            inp = self._linearize_triples_mask_rel(entry.data)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        for lex in entry.lexs:
            example = {
                "in" : inp,
                "out" : lex,
            }
            examples.append(example)

        return examples


    def process(self, split, out_dirname):
        """
        Processes and outputs training data for the model
        """ 
        output = {"data" : []}
        data = self.dataset.data[split]

        for i, entry in enumerate(data):
            examples = self.create_examples(entry)

            if split != "train":
                # keep just one example per tripleset
                examples = [examples[0]]

            for example in examples:
                output["data"].append(example)

        with open(os.path.join(out_dirname, f"{split}.json"), "w") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)


    def extract_refs(self, out_dirname, split):
        with open(f"{out_dirname}/{split}.ref", "w") as f:
            data = self.dataset.data[split]

            for entry in data:
                for lex in entry.lexs:
                    f.write(lex + "\n")
                f.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
        help="Name of the dataset to preprocess.")
    parser.add_argument("--dataset_dir", type=str, default=None,
        help="Path to the dataset")
    parser.add_argument("--mode", choices=["full", "desc_repl", "desc_cat", "mask_rel"], required=True,
        help="Preprocessing mode")
    parser.add_argument("--output", type=str, required=True,
        help="Name of the output directory")
    parser.add_argument('--splits', type=str, nargs='+', default=["train", "dev", "test"],
                    help='Dataset splits (e.g. train dev test)')
    parser.add_argument("--seed", type=int, default=42,
        help="Random seed.")
    parser.add_argument("--fewshot_splits", type=int, nargs='+', default=None,
        help="Generate few shot splits.")
    parser.add_argument("--extract_refs", action="store_true",
        help="Extract references for evaluation.")

    args = parser.parse_args()
    random.seed(args.seed)
    logger.info(args)
    dataset = get_dataset_class_by_name(args.dataset)()

    try:
        if args.fewshot_splits is not None:
            fewshot_splits = list(sorted(args.fewshot_splits))
            dataset.few_shot = True
            dataset.random_seed = args.seed
            dataset.max_fewshot_len = fewshot_splits[-1]
            
        dataset.load(splits=args.splits, path=args.dataset_dir)
    except FileNotFoundError as err:
        logger.error(f"Dataset could not be loaded")
        raise err
        
    try:
        out_dirname = args.output
        os.makedirs(out_dirname, exist_ok=True)
    except OSError as err:
        logger.error(f"Output directory {out_dirname} can not be created")
        raise err

    preprocessor = Preprocessor(
        dataset=dataset, 
        mode=args.mode
    )

    if args.extract_refs:
        for split in args.splits:
            preprocessor.extract_refs(
                out_dirname=out_dirname,
                split=split
            )
    elif args.fewshot_splits is not None:
        for fewshot_split in fewshot_splits:
            dataset.set_fewshot_split(fewshot_split)

            fewshot_out_dirname = os.path.join(out_dirname, str(args.seed), str(fewshot_split))
            os.makedirs(fewshot_out_dirname, exist_ok=True)

            preprocessor.process("train",
                out_dirname=fewshot_out_dirname
            )
            # dummy dev to satisfy dataloaders
            with open(os.path.join(fewshot_out_dirname, "dev.json"), "w") as f:
                json.dump({"data" : [{"in" : "", "out" : ""}]}, f)
    else:
        for split in args.splits:
            preprocessor.process(split, out_dirname=out_dirname)

    logger.info(f"Preprocessing finished.")
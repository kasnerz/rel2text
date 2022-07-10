#!/usr/bin/env python3

import json
import csv
import os
import logging
import re
import random
import datasets
import pandas as pd

from sentence_transformers import SentenceTransformer, util
from collections import defaultdict, namedtuple
from utils.tokenizer import Tokenizer

from datasets import load_dataset

logger = logging.getLogger(__name__)

RDFTriple = namedtuple('RDFTriple', ['head', 'rel', 'tail'])
RDFTripleDesc = namedtuple('RDFTripleDesc', ['head', 'head_desc', 'rel', 'rel_desc', 'tail', 'tail_desc'])

def get_dataset_class_by_name(name):
    """
    A helper function which allows to use the class attribute `name` of a Dataset 
    (sub)class as a command-line parameter for loading the dataset.
    """
    try:
        # case-insensitive
        available_classes = {o.name.lower() : o for o in globals().values() 
                                if type(o)==type(Dataset) and hasattr(o, "name")}
        return available_classes[name.lower()]
    except AttributeError:
        logger.error(f"Unknown dataset: '{args.dataset}'. Please create \
            a class with an attribute name='{args.dataset}' in 'data.py'.")
        return None


class DataEntry:
    def __init__(self, data, lexs, data_type="triples"):
        self.data = data
        self.lexs = lexs
        self.data_type = data_type

    def __repr__(self):
        return str(self.__dict__)


class Dataset:
    """
    Base class for the datasets
    """
    def __init__(self):
        self.data = {split: [] for split in ["train", "dev", "test"]}

    def load(self, splits, path=None):
        """
        Load the dataset. Path can be specified for loading from a directory
        or omitted if the dataset is loaded from HF.
        """
        raise NotImplementedError


class Rel2Text(Dataset):
    name="rel2text"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = Tokenizer()
        self.rels_filter = self.load_rels_filter()
        self.rels_accept = set()
        self.rels_emb = {}
        self.semsim_model = SentenceTransformer('all-distilroberta-v1')
        self.semsim_threshold = 0.9
        self.few_shot = False
        self.random_seed = None
        self.ids = defaultdict(list)


    def load_rels_filter(self):
        rels_filter = []

        with open("utils/relation_list/webnlg_v3_all.txt") as f:
            rels_filter += [x.strip() for x in f.readlines()]

        with open("utils/relation_list/kelm_all.txt") as f:
            rels_filter += [x.strip() for x in f.readlines()]

        rels_filter.sort()
        return rels_filter


    def precompute_semsim(self, rel_labels, rels_filter):
        embeddings1 = self.semsim_model.encode(rel_labels, convert_to_tensor=True, show_progress_bar=True)
        embeddings2 = self.semsim_model.encode(rels_filter, convert_to_tensor=True, show_progress_bar=True)
        scores = util.cos_sim(embeddings1, embeddings2)
        return scores


    def get_semsim(self, rel1, rel2, semsim_matrix):
        idx1 = self.rel_labels_idx[rel1]
        idx2 = self.rels_filter.index(rel2)
        return semsim_matrix[idx1][idx2].item()


    def has_no_similarities(self, rel, semsim_matrix):
        if rel in self.rels_accept:
            print("[OK]", rel)
            return True
        elif rel in self.rels_filter:
            print("[Reject-exact-match]", rel)
            return False
        else:
            highest_score = -1
            best_match = None

            for rel2 in self.rels_filter:
                score = self.get_semsim(rel, rel2, semsim_matrix)
                if score > highest_score:
                    highest_score = score
                    best_match = rel2
                if score >= self.semsim_threshold:
                    print("[Reject-semsim]", rel)
                    return False

            print(f"[Best-semsim] {highest_score:.3f} | {rel} -> {best_match}")
            self.rels_accept.add(rel)
            return True

    def create_entry(self, example):
        head = example["head"]
        head_desc = example["head_desc"]
        rel = example["relation"]
        rel_desc = example["rel_desc"]
        tail = example["tail"]
        tail_desc = example["tail_desc"]

        triples = [RDFTripleDesc(head, head_desc, rel, rel_desc, tail, tail_desc)]
        lexs = [example["response"]]

        entry = DataEntry(data=triples, lexs=lexs)
        return entry


    def assert_no_test_bias(self):
        test_rel_labels =list(set([x.data[0].rel for x in self.data["test"]]))
        train_rel_labels =list(set([x.data[0].rel for x in self.data["train"] + self.data["dev"]]))
        posthoc_semsim_matrix = self.precompute_semsim(
            rel_labels=test_rel_labels, 
            rels_filter=train_rel_labels
        )
        for idx1, rel_test in enumerate(test_rel_labels):
            highest_score = -1
            best_match = None
            for idx2, rel_train in enumerate(train_rel_labels):
                score = posthoc_semsim_matrix[idx1][idx2].item()

                if score > highest_score:
                    highest_score = score
                    best_match = rel_train

            if highest_score >= self.semsim_threshold:
                raise ValueError(f"[Best-semsim] {highest_score:.3f} | {rel_test} -> {best_match}")


    def set_fewshot_split(self, fewshot_split):
        self.data["train"] = self.limited_train_data[:fewshot_split]


    def load(self, path, splits):
        rel_num = 0
        rel_prev = None
        id_prev = -1

        examples_clean = []
        examples_biased = []

        data = pd.read_csv(os.path.join(path, "rel2text_raw_annotated.tsv"), sep='\t') 
        rel_labels = list(set(data["relation"].to_list()))
        self.rel_labels_idx = dict([(y,x) for x,y in enumerate(rel_labels)])

        semsim_matrix = self.precompute_semsim(
            rel_labels=rel_labels, 
            rels_filter=list(self.rels_filter)
        )
        logger.info("Similarities precomputed")

        for i, example in data.iterrows():
            if example["state"] != "ok":
                continue

            # keep only one (successful) human reference for each example
            if example["id"] != id_prev:
                id_prev = example["id"]
            else:
                continue

            rel = example["relation"]
            if rel != rel_prev:
                rel_prev = rel
                rel_num += 1

            if self.has_no_similarities(example["relation"], semsim_matrix):
                examples_clean.append(example)
            else:
                examples_biased.append(example)

        examples_total = len(examples_clean) + len(examples_biased)
        examples_test_max = int(0.15 * examples_total)
        rel_prev = None

        for i, example in enumerate(examples_clean):
            entry = self.create_entry(example)

            if example["relation"] != rel_prev:
                rel_prev = example["relation"]

                if len(self.data["test"]) > examples_test_max:
                    break
            
            self.data["test"].append(entry)
            self.ids["test"].append(example["id"])


        rel_prev = None
        split = None

        for j, example in enumerate(examples_clean[i:] + examples_biased):
            entry = self.create_entry(example)

            if example["relation"] != rel_prev:
                rel_prev = example["relation"]

                if j % 10 == 0:
                    split = "dev"
                else:
                    split = "train"

            self.data[split].append(entry)
            self.ids[split].append(example["id"])

        self.assert_no_test_bias()
        logger.info([(split, len(self.data[split])) for split in ["train", "dev", "test"]])


        if self.few_shot:
            rels_used = set()
            self.limited_train_data = []
            random.seed(self.random_seed)
            random.shuffle(self.data["train"])

            for example in self.data["train"]:
                rel = example.data[0].rel
                if rel in rels_used:
                    continue
                else:
                    rels_used.add(rel)
                self.limited_train_data.append(example)

                if len(self.limited_train_data) == self.max_fewshot_len:
                    break


class WebNLG_v3(Dataset):
    name="webnlg_v3"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = Tokenizer()


    def load(self, path, splits):
        # GEM-based loading
        dataset = load_dataset("gem", "web_nlg_en")

        for split in splits:
            data = dataset[split if split != "dev" else "validation"]

            for example in data:
                triples = example["input"]
                triples = [t.split("|") for t in triples]
                triples = [[self.tokenizer.normalize(text, remove_quotes=True, remove_parentheses=True) 
                            for text in t] 
                                for t in triples]
                triples = [RDFTriple(*t) for t in triples]

                if split == "test":
                    lexs = example["references"]
                else:
                    lexs = [example["target"]]

                entry = DataEntry(data=triples, lexs=lexs)
                self.data[split].append(entry)



class KeLM(Dataset):
    name="kelm"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = Tokenizer()


    def normalize(self, text):
        text = re.sub(r"\+(\d*)", r"\1", text)

        # text = self.tokenizer.detokenize(text)
        # detokenization is too slow, this is a simpler workaround
        text = text.strip()
        text = text.replace("( ", "(")
        text = text.replace(" )", ")")
        text = text.replace(" ,", ",")
        text = text.replace(" 's", "'s")
        text = text.replace(" : ", ":")
        text = text.replace("--", "-")
        text = re.sub(r"(\s)\s+", r"\1", text)

        return text


    def load(self, path, splits):
        entries = 0
        skipped = 0
        damaged = 0
        rel_idx = 0
        nr_triples = defaultdict(int)
        train_rels = set()
        dev_rels = set()

        with open(os.path.join(path, "kelm_generated_corpus.jsonl")) as f:
            for line in f.readlines():
                try:
                    j = json.loads(line)
                    entries += 1

                    # !!!! DEBUG !!!!
                    # if entries > 10000:
                    #     break

                    nr_triples[len(j["triples"])]+=1
                    lexs = [self.normalize(j["gen_sentence"])]

                    # skip damaged examples
                    if "⁇" in lexs[0]:
                        damaged += 1
                        continue

                    triples = []
                    skip = False

                    for t in j["triples"]:
                        rel_label = t[1]

                        # skip examples which are not a proper RDF triple or contain a mislabeled relation
                        if len(t) != 3 or not rel_label[0].islower():
                            skip = True
                            break

                        if rel_label in train_rels:
                            split = "train"
                        elif rel_label in dev_rels:
                            split = "dev"
                        else:
                            rel_idx += 1
                            if rel_idx % 100 == 0:
                                dev_rels.add(rel_label)
                                split = "dev"
                            else:
                                train_rels.add(rel_label)
                                split = "train"

                        t_norm = [self.normalize(t[0]), self.normalize(t[1]), self.normalize(t[2])]

                        triple = RDFTriple(*t_norm)
                        triples.append(triple)

                    if skip == True:
                        continue

                    entry = DataEntry(data=triples, lexs=lexs)
                    self.data[split].append(entry)

                    if entries % 10000 == 0:
                        logger.info(entry)
                        logger.info(f"{entries - skipped - damaged}/{entries} loaded ({skipped} skipped, {damaged} damaged)")
                        logger.info(f"# of triples: " + str(nr_triples))
                except KeyError:
                    skipped += 1
                    pass
                except Exception as e:
                    raise

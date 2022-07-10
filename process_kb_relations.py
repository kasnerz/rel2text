#!/usr/bin/env python3

import os
import argparse
import logging
import json
import random
import pandas as pd
import re
import nltk
import stanza

from datetime import datetime, timezone


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def get_str(ent):
    if "name" in ent:
        return ent["name"]
    elif "value" in ent:
        return ent["value"]
    else:
        raise ValueError(ent)

def normalize(s, split_camelcase=False):
    # remove underscores
    s = re.sub(r'_', r' ', s)
    s = re.sub(r'(\s)\s+', r'\1', s)

    s = re.sub(r'"', r'', s)
    s = re.sub(r'``', r'', s)
    s = re.sub(r"''", r'', s)

    s = re.sub(r'\(.*\)', r'', s)
    s = s.strip()

    if split_camelcase:
        # split basic camel case, lowercase first letters

        # the for loop is a hack to condition on lowercase letters after the expression (group(3)) 
        # and split two-letter clusters at the same time
        for x in range(3):
            s = re.sub(r"([a-z1-9])([A-Z])([a-z])",
                lambda m: rf"{m.group(1)} {m.group(2).lower()}{m.group(3)}", s)

    # make ISO dates more readable
    iso_date = re.match(r"-*(\d+)-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", s)
    if iso_date:
        year = iso_date.group(1)

        if s.startswith("-"):
            s = f"{year} BC"
        else:
            iso_date = datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")

            if iso_date.month == 1 and iso_date.day == 1:
                s = f"{iso_date.year}"

            elif iso_date.hour == 0 and iso_date.minute == 0:
                s = iso_date.strftime('%Y-%m-%d')
            else:
                s = iso_date.strftime('%Y-%m-%d %H:%M:%S')

    return s


def normalize_desc(desc, nlp):
    if desc is not None:
        desc = desc.split("\n")[0]

        # replace markdown links with names only
        desc = re.sub(r'\[([\w\d]+)\]\(.*\)', r'\1', desc)

        if len(desc) >= 256:
            desc = desc[:256]
        
        doc = nlp(desc)
        sents = [sentence.text for sentence in doc.sentences]
        desc = sents[0]
        desc = re.sub(r'[^A-Za-z0-9 \.\(\)-–—@:,]', '', desc)

    return desc

def skip_relation(rel_name, rel_desc):
    if rel_desc is not None and "Reserved for DBpedia" in rel_desc:
        return True

    if re.search(r"\bid\b", rel_name, re.IGNORECASE) \
        or re.search(r"ID", rel_name) \
        or re.search(r"number$", rel_name, re.IGNORECASE) \
        or re.search(r"\bcode\b", rel_name, re.IGNORECASE) \
        or "identifier" in rel_desc:
        # or re.search(r"wikidata", rel_name, re.IGNORECASE):
        return True

    return False

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def looks_like_hash_or_id(s):
    # nr_of_digits = sum(c.isdigit() for c in s)
    # nr_of_letters = sum(c.isalpha() for c in s)
    nr_of_spaces = sum(c.isspace() for c in s)

    # dl_ratio = (nr_of_digits / (nr_of_letters+1))
    if len(s) > 20 and nr_of_spaces == 0:
        print("H-id", s)
        return True

    return False

def has_unwanted_chars(s):
    if re.search(r'[^A-Za-z0-9 \-\.–—:\u00C0-\u00FF\u0100-\u024F\'\+\!\&\,]+', s):
        print("U-ch", s)
        return True
    return False

def has_multiple_numbers(s):
    if re.match(r'\d+\.*\d*\s*,', s):
        print("M-nm", s)
        return True

    return False


def skip_ent(ent_name, ent_desc):
    if (len(ent_name) > 64
        or ent_name.startswith("http")
        or "XMLSchema#" in ent_name
        or "Category:" in ent_name
        or looks_like_hash_or_id(ent_name)
        or has_multiple_numbers(ent_name)
        or has_unwanted_chars(ent_name) # chemical formulas, chinese chars...
        ):
        return True

    return False

def skip_ent_pair(head_name, tail_name):
    head_name = head_name.lower()
    tail_name = tail_name.lower()

    if (head_name in tail_name) or (tail_name in head_name):
        return True 

def create_crowdsourcing_csv(out_file, kbs, nlp):
    csv_columns = ["id", "kb", "rel_uri", "relation", "head", "tail", "rel_desc", "head_desc", "tail_desc"]
    csv = {c : [] for c in csv_columns}

    duplicates = set()

    example_id = 0
    skipped_relations = 0
    skipped_examples = 0

    for kb in kbs:
        filename = f"data/kb/{kb}/output.json"

        with open(filename) as f:
            j = json.load(f)

            for rel in j:
                rel_name = rel["name"]
                rel_uri = rel["uri"]
                rel_name = normalize(rel_name, split_camelcase=True)
                rel_desc = rel.get("desc")
                rel_desc = normalize_desc(rel_desc, nlp=nlp)

                if rel_name in csv["relation"]:
                    duplicates.add(rel_name)
                    continue

                if skip_relation(rel_name, rel_desc):
                    skipped_relations += 1
                    continue

                # random_example = random.choice(rel["examples"])

                for random_example in rel["examples"]:
                    head = random_example["head"]
                    tail = random_example["tail"]
                    head_desc = head.get("desc")
                    tail_desc = tail.get("desc")

                    head_name = normalize(get_str(head))
                    tail_name = normalize(get_str(tail))
                    head_desc = normalize_desc(head_desc, nlp=nlp)
                    tail_desc = normalize_desc(tail_desc, nlp=nlp)

                    if skip_ent(head_name, head_desc) \
                        or skip_ent(tail_name, tail_desc) \
                        or skip_ent_pair(head_name, tail_name):
                        skipped_examples += 1
                        continue

                    csv["id"].append(example_id)
                    csv["kb"].append(kb)
                    csv["rel_uri"].append(rel_uri)
                    csv["relation"].append(rel_name)
                    csv["head"].append(head_name)
                    csv["tail"].append(tail_name)
                    csv["rel_desc"].append(rel_desc)
                    csv["head_desc"].append(head_desc)
                    csv["tail_desc"].append(tail_desc)

                    example_id += 1

                    if example_id % 1000 == 0:
                        logger.info(f"{example_id} examples processed.")
                        logger.info("Skipped relations: " + str(skipped_relations))
                        logger.info("Skipped examples: " + str(skipped_examples))

                # logger.info(f"{rel_name}: ({head_name} -> {tail_name}) | {rel_desc}")

    logger.info("Duplicates: " + str(duplicates))
    logger.info("Skipped relations: " + str(skipped_relations))
    logger.info("Skipped examples: " + str(skipped_examples))

    df = pd.DataFrame(csv)
    # df = df.sort_values("relation")

    df.to_csv(out_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_file", type=str, required=True,
        help="Output file")
    parser.add_argument("-k", "--kbs", nargs="+", type=str, default=["yago", "dbpedia", "wikidata"],
        help="Knowledge bases to use")

    args = parser.parse_args()
    logger.info(args)
    random.seed(42)

    nlp = stanza.Pipeline(lang='en', processors='tokenize')
    create_crowdsourcing_csv(out_file=args.out_file, kbs=args.kbs, nlp=nlp)
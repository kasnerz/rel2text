#!/usr/bin/env python3

import os
import argparse
import logging
import json
import time
import requests
import random
import time

from collections import defaultdict
# from rdflib.graph import Graph

from SPARQLWrapper import SPARQLWrapper, JSON
from requests.structures import CaseInsensitiveDict

from bs4 import BeautifulSoup

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


class Relation:
    def __init__(self, uri, name=None, desc=None, aliases=[]):
        self.uri = uri
        self.name = name
        self.desc = desc
        self.aliases = aliases

    def __str__(self):
        return f"Relation({self.name})"

    def to_json(self):
        return self.__dict__

    def __eq__(self, o):
        return self.uri == o.uri

    def __hash__(self):
        return hash(self.uri)

class Entity:
    def __init__(self, uri, name=None, desc=None):
        self.uri = uri
        self.name = name
        self.desc = desc
        self.datatype = "entity"

    def __str__(self):
        return f"Entity({self.name} - {self.desc})"

    def to_json(self):
        return self.__dict__


class Literal:
    def __init__(self, datatype, value):
        self.datatype = datatype
        self.value = value

    def __str__(self):
        return f"{self.datatype}({self.value})"

    def to_json(self):
        return self.__dict__


class Extractor:
    def __init__(self, args):
        self.results = defaultdict(list)
        self.fetch_descriptions = not args.no_descriptions
        self.nr_of_entities = args.entities
        self.start_rel_id = args.start_rel_id
        self.end_rel_id = args.end_rel_id
        self.user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"
        
    def save_to_json(self, out_file):
        output = [ 
            {
                "uri" : key.uri,
                "name" : key.name,
                "desc" : key.desc,
                "aliases" : key.aliases,
                "examples" : [{"head" : v[0].to_json(), "tail" : v[1].to_json()} for v in value]
            }
            for key, value in self.results.items()
        ]

        with open(os.path.join(out_file), "w") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)


class WikidataExtractor(Extractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ent_cache = {}
        self.ldf_url = "https://query.wikidata.org/bigdata/ldf"
        self.api_url = "https://www.wikidata.org/w/api.php"

    def _ldf_query(self, rel_id, page=1):
        # time.sleep(0.5)

        def _call():
            params = {
                "subject" : "",
                "predicate" : f"wdt:{rel_id}",
                "object" : "",
                "page" : page
            }
            headers = CaseInsensitiveDict()
            headers["Accept"] = "application/ld+json"
            headers["User-Agent"] = self.user_agent
            resp = requests.get(self.ldf_url, params=params, headers=headers)
            return resp

        for x in range(5):
            resp = _call()
            if resp.status_code == 429:
                time.sleep(10)
            else:
                try:
                    j = resp.json()
                    graph = j["@graph"]
                    return graph
                except:
                    print(resp)
                    break
        return []


    def extract_kb(self):
        ret = self._load_and_cache_properties()

        for r in ret["results"]["bindings"][self.start_rel_id:min(self.end_rel_id,len(ret["results"]["bindings"]))]:
            try:
                uri = r["property"]["value"]
                name = r["propertyLabel"]["value"]
                desc = r["propertyDescription"]["value"]
                aliases = r["altLabel_list"]["value"].split(";")

                rel = Relation(uri=uri, name=name, desc=desc, aliases=aliases)
                rel_id = uri.split("/")[-1]
                
                logger.info("---------------")
                logger.info(rel)

                # a hacky way to get at least a somehow random sample of entities from Wikidata LDF
                graph = self._ldf_query(rel_id, page=1)

                if not graph: 
                    continue

                total_items = [i for i in graph if "http://www.w3.org/ns/hydra/core#totalItems" in i][0]
                total_items = total_items["http://www.w3.org/ns/hydra/core#totalItems"]
                logger.info(f"Total items: {total_items}")
                ldf_items_per_page = 100
                pages_to_retrieve = 4
                pages_total = int(total_items / ldf_items_per_page)

                if pages_total <= 1:
                    continue

                random_page_numbers = random.sample(range(1,pages_total), min(pages_total-1, pages_to_retrieve))

                for page_nr in random_page_numbers:
                    graph += self._ldf_query(rel_id, page=page_nr)

                logger.info(f"Retrieved items: {len(graph)}")
                random.shuffle(graph)

                for e_id, e in enumerate(graph):
                    if not rel_id in e:
                        continue

                    subj_id = e["@id"]
                    obj_id = e[rel_id]

                    subj = self._extract_entity(subj_id)
                    obj = self._extract_entity(obj_id)

                    if subj is None or obj is None:
                        continue

                    logger.info(str(subj) + ", " + str(obj))
                    self.results[rel].append((subj, obj))

                    if len(self.results[rel]) >= self.nr_of_entities:
                        break
            except Exception as e:
                logger.exception(e)
                time.sleep(120)

    def _get_entity_desc(self, ent_id):
        params = {
            "action": "wbgetentities",
            "ids" : f"{ent_id}",
            "languages" : "en",
            "props" : "descriptions",
            "format" : "json"
        }
        try:
            headers = CaseInsensitiveDict()
            headers["User-Agent"] = self.user_agent
            r = requests.get(url=self.api_url, params=params, headers=headers)
            j = r.json()
            desc = j["entities"][ent_id]["descriptions"]["en"]["value"]
            return desc
        except:
            logger.warning(f"Cannot fetch description for entity {ent_id}")
            return None


    def _extract_entity(self, ent_obj):
        if type(ent_obj) is str and ent_obj.startswith("wd:"):
            ent_id = ent_obj.split(":")[-1]
            ent_name = self._get_label_by_id(ent_id)
            if ent_name is None:
                return None

            ent_uri = f"http://www.wikidata.org/entity/{ent_id}"
            desc = self._get_entity_desc(ent_id) 
            ent = Entity(uri=ent_uri, name=ent_name, desc=desc)
        elif type(ent_obj) is str:
            datatype = "plain"
            value = ent_obj
            ent = Literal(datatype=datatype, value=value)

        elif type(ent_obj) is list and all([type(x) is str for x in ent_obj]):
            # list of entities, too expensive to resolve -> skip
            if any([x.startswith("wd:") for x in ent_obj]):
                return None

            datatype = "plain"
            value = ", ".join(ent_obj)
            ent = Literal(datatype=datatype, value=value)
        else:
            logger.warning(f"Unknown entity: {ent_obj}")
            return None
        
        return ent
                

    def _get_label_by_id(self, wikidata_id):
        if wikidata_id in self.ent_cache:
            return self.ent_cache[wikidata_id]

        api_url = "https://www.wikidata.org/w/api.php"
        api_params = {
            "action": "wbgetentities",
            "ids" : wikidata_id,
            "languages" : "en",
            "props" : "labels",
            "format" : "json"
        }
        headers = CaseInsensitiveDict()
        headers["User-Agent"] = self.user_agent

        r = requests.get(url=api_url, params=api_params, headers=headers)
        j = r.json()
        try:
            ent = j["entities"][wikidata_id]["labels"]["en"]["value"]
            self.ent_cache[wikidata_id] = ent
            return ent
        except:
            return None


    def _load_and_cache_properties(self):
        prop_cache_file = "data/kb/wikidata/properties.json"

        if os.path.isfile(prop_cache_file):
            with open(prop_cache_file) as f:
                properties = json.load(f)
                return properties
        else:
            # Wikidata has strict limits on SPARQL queries (1 per minute), should be used with care

            # alternative URLs with indepedent timeout (or so it seems)
            # sparql = SPARQLWrapper("https://query.wikidata.org/bigdata/namespace/wdq/sparql")
            sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
            sparql.setReturnFormat(JSON)

            sparql.setQuery("""
                SELECT ?property ?propertyLabel ?propertyDescription (GROUP_CONCAT(DISTINCT(?altLabel); separator = ";") AS ?altLabel_list) WHERE {
                    ?property a wikibase:Property .
                    OPTIONAL { ?property skos:altLabel ?altLabel . FILTER (lang(?altLabel) = "en") }
                    SERVICE wikibase:label { bd:serviceParam wikibase:language "en" .}
                 }
                GROUP BY ?property ?propertyLabel ?propertyDescription
                """
            )
            ret = sparql.queryAndConvert()
            os.makedirs(os.path.dirname(prop_cache_file), exist_ok=True)

            with open(prop_cache_file, "w") as f:
                json.dump(ret, f, indent=4, ensure_ascii=False)
    
            return ret   



class DBPediaExtractor(Extractor):
    def _extract_entity(self, ent_obj, ent_desc=None):
        ent_type = ent_obj["type"]

        if ent_type == "uri":
            uri = ent_obj["value"]
            name = uri.split("/")[-1]

            if name.startswith("Wikisource:"):
                return None

            desc = self._get_entity_desc(uri)
            # if ent_desc and ent_desc.get("value"):
            #     desc = ent_desc["value"]
            # else:
            #     desc = None

            ent = Entity(uri=uri, name=name, desc=desc)
        elif ent_type == "typed-literal":
            datatype =  ent_obj["datatype"]
            value = ent_obj["value"]
            ent = Literal(datatype=datatype, value=value)

        elif ent_type == "literal":
            datatype = "plain"
            value = ent_obj["value"]
            ent = Literal(datatype=datatype, value=value)
        else:
            logger.warning(f"Undefined type: {ent_type}")
            return None
        
        return ent


    def _get_entity_desc(self, uri):
        r = requests.get(uri)
        b = BeautifulSoup(r.text, 'lxml')
        abstract = b.findAll("span", {"property" : "dbo:abstract", "lang" : "en"})

        if len(abstract) > 0:
            desc = abstract[0].text
            return desc

        return None


    def extract_kb(self):
        sparql = SPARQLWrapper("http://dbpedia.org/sparql")
        sparql.setReturnFormat(JSON)

        sparql.setQuery("""
            SELECT ?property  ?comment where {
             ?property a rdf:Property .
             ?property rdfs:comment ?comment .
            FILTER (lang(?comment) = 'en')
            }
            """
        )
        ret = sparql.queryAndConvert()

        for r in ret["results"]["bindings"][self.start_rel_id:self.end_rel_id]:
            try:
                uri = r["property"]["value"]
                name = uri.split("/")[-1]
                desc = r["comment"]["value"]

                rel = Relation(uri=uri, name=name, desc=desc)

                if rel in self.results:
                    continue

                logger.info(rel)

                query = f"""
                    SELECT ?subject ?object WHERE {{
                      ?subject <{uri}> ?object
                    }}
                    ORDER BY RAND()
                    LIMIT {self.nr_of_entities}
                    """
                sparql.setQuery(query)
                examples = sparql.queryAndConvert()

                for e in examples["results"]["bindings"]:
                    subj = self._extract_entity(e["subject"])
                    obj = self._extract_entity(e["object"])

                    logger.info(str(subj) + ", " + str(obj))

                    if subj is None or obj is None:
                        continue
                    
                    self.results[rel].append((subj, obj))
            except Exception as e:
                logger.exception(e)
                time.sleep(120)


class YagoExtractor(Extractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.yago_kb_path = "data/kb/yago/yago-wd-facts.nt"
        self.schema_org_path = "data/kb/yago/schemaorg-current-https.jsonld"

    def _get_entity_desc(self, uri):

        if uri.startswith("http://yago-knowledge.org/resource/"):
            try:
                r = requests.get(uri)
                b = BeautifulSoup(r.text, 'lxml')
                a = b.find_all("a", string="rdfs:comment")

                if len(a) > 0:
                    descs = a[0].parent.next_sibling.findAll("span", {"lang" : "en"})
                    if len(descs) > 0:
                        desc = descs[0].text
                        return desc
            except Exception as e:
                logger.warning(e)

        return None


    def extract_kb(self):
        prop_desc = {}

        with open(self.schema_org_path) as sf:
            j = json.load(sf)
            graph = j["@graph"]
            props = [x for x in graph if x["@type"] == "rdf:Property"]

            for prop in props:
                name = prop["rdfs:label"]
                desc = prop["rdfs:comment"]

                if type(desc) is not str:
                    continue

                prop_desc[name] = desc
        
        rel_to_ents = defaultdict(list)

        with open(self.yago_kb_path) as f:
            logger.info("Fetching all relations in the KG (this may take a while)...")
            for line_nr, line in enumerate(f.readlines()):
                if line_nr % 100000 == 0:
                    logger.info(f"{line_nr} lines processed")

                items = [x.replace("<", "").replace(">", "") for x in line.split("\t")]
                subj_uri, rel_uri, obj_uri = items[:3]
                rel_name = rel_uri.split("/")[-1]

                if rel_name not in prop_desc:
                    continue
                rel_to_ents[rel_uri].append((subj_uri, obj_uri))
            
        logger.info(f"Relations extracted: {len(rel_to_ents.keys())}")

        for idx, (rel_uri, list_of_ents) in enumerate(rel_to_ents.items()):

            if idx < self.start_rel_id:
                continue
            elif idx >= self.end_rel_id:
                break
            try:
                sample = random.sample(list_of_ents, min(self.nr_of_entities,len(list_of_ents)))
                rel_name = rel_uri.split("/")[-1]
                desc = prop_desc[rel_name]
                rel = Relation(uri=rel_uri, name=rel_name, desc=desc)
                logger.info(rel) 

                for subj_uri, obj_uri in sample:
                    subj_desc = self._get_entity_desc(subj_uri)
                    obj_desc = self._get_entity_desc(obj_uri)

                    subj_name = subj_uri.split("/")[-1]
                    subj = Entity(uri=subj_uri, name=subj_name, desc=subj_desc)

                    obj_name = obj_uri.split("/")[-1]
                    obj = Entity(uri=obj_uri, name=obj_name, desc=obj_desc)

                    logger.info(str(subj) + ", " + str(obj))
                    self.results[rel].append([subj, obj])
            except Exception as e:
                logger.exception(e)
                time.sleep(120)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--kb", type=str, required=True,
        help="Knowledge base to extract relations from: ['yago', 'dbpedia', 'wikidata']")
    parser.add_argument("-o", "--out_file", type=str, required=True,
        help="Output file")
    parser.add_argument("-e", "--entities", type=int, default=5,
        help="Number of sampled entity pairs for each relation.")
    parser.add_argument("-n", "--no_descriptions", action="store_true",
        help="Do not fetch entity descriptions (may speed up the process considerably.")
    parser.add_argument("--start_rel_id", type=int, default=None,
        help="Start from n-th relation.")
    parser.add_argument("--end_rel_id", type=int, default=None,
        help="End with n-th relation.")

    args = parser.parse_args()
    logger.info(args)
    random.seed(42)

    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    kb = args.kb
    if kb == "yago":
        e = YagoExtractor(args=args)
    elif kb == "dbpedia":
        e = DBPediaExtractor(args=args)
    elif kb == "wikidata":
        e = WikidataExtractor(args=args)
    else:
        raise ValueError("Unknown KB")

    e.extract_kb()
    e.save_to_json(args.out_file)

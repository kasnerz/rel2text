from nubia_score import Nubia
import numpy as np
from .metric import ReferencedMetric
from logzero import logger

import pprint

class NUBIA(ReferencedMetric):
    def __init__(self):
        """Downloads pretrained models for nubia, and loads them into memory"""
        self.metric = Nubia()

    def compute(self, predictions, references):
        """Run Nubia"""
        scores = []
        semsim = []
        contr = []
        neutr = []
        agree = []
        perpl = []

        is_multiref = isinstance(references.untokenized[0], list) and any([len(refs) > 1 for refs in references.untokenized])

        for ref, pred in zip(references.untokenized, predictions.untokenized):
            if is_multiref:
                # For multi-reference data, compute micro-average.
                mutli_scores = []
                for _ref in ref:
                    mutli_scores.append(self.metric.score(_ref, pred))
                scores.append(np.mean(mutli_scores))
            else:
                if isinstance(ref, list):
                    ref = ref[0]

                logger.info("--------------------")
                logger.info("pred:" + str(pred))
                logger.info("ref:" + str(ref))
                res = self.metric.score(ref, pred, get_features=True)
                logger.info(pprint.pformat(res))

                scores.append(res["nubia_score"])
                semsim.append(res["features"]["semantic_relation"])
                contr.append(res["features"]["contradiction"])
                neutr.append(res["features"]["irrelevancy"])
                agree.append(res["features"]["logical_agreement"])
                perpl.append(res["features"]["grammar_hyp"])


        return {
            "nubia": np.mean(scores),
            "nubia_semsim": np.mean(semsim),
            "nubia_contr": np.mean(contr),
            "nubia_neutr": np.mean(neutr),
            "nubia_agree": np.mean(agree),
            "nubia_perpl": np.mean(perpl),
        }
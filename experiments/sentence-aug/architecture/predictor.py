from typing import Iterator, List, Dict

from allennlp.common import JsonDict
from allennlp.predictors import Predictor
from overrides import overrides

import numpy as np


@Predictor.register("exercise_classifier")
class ExerciseClassifierPredictor(Predictor):
    def predict(self, sent: str) -> JsonDict:
        return self.predict_json({'sent': sent})

    def _json_to_instance(self, json_dict: JsonDict) -> JsonDict:
        sent = json_dict['sent']
        return self._dataset_reader.text_to_instance(sent)
    
    def dump_line(self, outputs: JsonDict) -> str:
        max_prob = max(outputs['probs'])
        return [(self._model.vocab.get_token_from_index(label_id, 'labels'), prob)
                for label_id, prob in enumerate(outputs['probs']) if prob == max_prob]

@Predictor.register("topk_exercise_classifier")
class ExerciseClassifierPredictor(Predictor):
    def predict(self, sent: str) -> JsonDict:
        return self.predict_json({'sent': sent})

    def _json_to_instance(self, json_dict: JsonDict) -> JsonDict:
        sent = json_dict['sent']
        return self._dataset_reader.text_to_instance(sent)
    
    def dump_line(self, outputs: JsonDict, k: int = 1) -> str:
        probs = np.array(outputs['probs'])
        topk_i = np.argpartition(probs, len(probs) - k)[-k:][::-1]
        return [(self._model.vocab.get_token_from_index(i, 'labels'), probs[i])
                for i in topk_i]
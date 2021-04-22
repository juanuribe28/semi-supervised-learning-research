from typing import Iterator, List, Dict

import torch

from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder

@Model.register('universal_dual_classifier')
class Net(Model):
    
    def __init__(self, 
                 vocab: Vocabulary,
                 sentence_embedder: TextFieldEmbedder,
                 verb_embedder: TextFieldEmbedder,
                 sentence_encoder: Seq2VecEncoder, 
                 verb_encoder: Seq2VecEncoder, 
                 sentence_dropout: float = 0,
                 verb_dropout: float = 0,
                 sentence_weight: float = 0.5) -> None:
        super().__init__(vocab)
        
        num_labels = vocab.get_vocab_size("labels")

        if sentence_weight != 0:
            self.s_embedder = sentence_embedder
            self.s_encoder = sentence_encoder
            self.s_linear = torch.nn.Linear(in_features = self.s_encoder.get_output_dim(), 
                                               out_features = num_labels)
            self.s_dropout = torch.nn.Dropout(sentence_dropout)

        if sentence_weight != 1:
            self.v_embedder = verb_embedder
            self.v_encoder = verb_encoder
            self.v_linear = torch.nn.Linear(in_features = self.v_encoder.get_output_dim(), 
                                                    out_features = num_labels)
            self.v_dropout = torch.nn.Dropout(verb_dropout)

        self.s_weight = sentence_weight
        self.v_weight = 1 - sentence_weight 
        
        self.accuracy = CategoricalAccuracy()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, 
                sentence: Dict[str, torch.Tensor],
                verb: Dict[str, torch.Tensor], 
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        
        output = dict()

        if self.s_weight != 0:
            # Shape: (batch_size, num_tokens, s_embedding_dim)
            s_embedded_text = self.s_dropout(self.s_embedder(sentence))
            # Shape: (batch_size, num_tokens)
            s_mask = get_text_field_mask(sentence)
            # Shape: (batch_size, s_encoding_dim)
            s_encoded_text = self.s_dropout(self.s_encoder(s_embedded_text, s_mask))
            # Shape: (batch_size, num_labels)
            s_logits = self.s_linear(s_encoded_text)
            output['s_logits'] = s_logits

        if self.v_weight != 0:
            # Shape: (batch_size, num_tokens, v_embedding_dim)
            v_embedded_text = self.v_dropout(self.v_embedder(sentence))
            # Shape: (batch_size, num_tokens)
            v_mask = get_text_field_mask(sentence)
            # Shape: (batch_size, v_encoding_dim)
            v_encoded_text = self.v_dropout(self.v_encoder(v_embedded_text, v_mask))
            # Shape: (batch_size, num_labels)
            v_logits = self.v_linear(v_encoded_text)
            output['v_logits'] = v_logits
        
        if self.s_weight == 0:
            logits = v_logits
        elif self.v_weight == 0:
            logits = s_logits
        else:
            logits = s_logits * self.s_weight + v_logits * self.v_weight

        output['logits'] = logits
        output['probs'] = torch.nn.functional.softmax(logits, dim=-1)
        
        if label is not None:
            self.accuracy(logits, label)
            output['loss'] = self.loss(logits, label)
        return output
        
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self.accuracy.get_metric(reset)}

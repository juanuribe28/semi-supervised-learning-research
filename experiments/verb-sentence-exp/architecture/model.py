from typing import Iterator, List, Dict

import torch

from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder

@Model.register('dual_simple_classifier')
class Net(Model):
    
    def __init__(self, 
                 vocab: Vocabulary, 
                 sentence_embedder: TextFieldEmbedder,
                 verb_embedder: TextFieldEmbedder,
                 sentence_encoder: Seq2VecEncoder, 
                 verb_encoder: Seq2VecEncoder, 
                 sentence_dropout: float = 0,
                 verb_dropout: float = 0) -> None:
        super().__init__(vocab)
        self.s_embedder = sentence_embedder
        self.s_encoder = sentence_encoder
        self.v_embedder = verb_embedder
        self.v_encoder = verb_encoder
        num_labels = vocab.get_vocab_size("labels")
        self.s_linear = torch.nn.Linear(in_features = self.s_encoder.get_output_dim(), 
                                               out_features = num_labels)
        self.v_linear = torch.nn.Linear(in_features = self.v_encoder.get_output_dim(), 
                                               out_features = num_labels)
        self.s_dropout = torch.nn.Dropout(sentence_dropout)
        self.v_dropout = torch.nn.Dropout(verb_dropout)

        self.merge = torch.nn.Bilinear(in1_features = num_labels,
                                       in2_features = num_labels,
                                       out_features = num_labels)

        self.accuracy = CategoricalAccuracy()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, 
                sentence: Dict[str, torch.Tensor],
                verb: Dict[str, torch.Tensor], 
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, s_embedding_dim)
        s_embedded_text = self.s_dropout(self.s_embedder(sentence))
        # Shape: (batch_size, num_tokens)
        s_mask = get_text_field_mask(sentence)
        # Shape: (batch_size, s_encoding_dim)
        s_encoded_text = self.s_dropout(self.s_encoder(s_embedded_text, s_mask))
        # Shape: (batch_size, num_labels)
        s_logits = self.s_linear(s_encoded_text)

        # Shape: (batch_size, num_tokens, v_embedding_dim)
        v_embedded_text = self.v_dropout(self.v_embedder(sentence))
        # Shape: (batch_size, num_tokens)
        v_mask = get_text_field_mask(sentence)
        # Shape: (batch_size, v_encoding_dim)
        v_encoded_text = self.v_dropout(self.v_encoder(v_embedded_text, v_mask))
        # Shape: (batch_size, num_labels)
        v_logits = self.v_linear(v_encoded_text)

        # Shape: (batch_size, num_labels)
        logits = self.merge(s_logits, v_logits)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        output = {'s_logits':s_logits, 'v_logits':v_logits, 'logits':logits, 'probs':probs}
        
        if label is not None:
            self.accuracy(logits, label)
            output['loss'] = self.loss(logits, label)
        return output
        
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self.accuracy.get_metric(reset)}

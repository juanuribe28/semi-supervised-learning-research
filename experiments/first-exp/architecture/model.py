from typing import Iterator, List, Dict

import torch

from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.nn.util import get_text_field_mask
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder

@Model.register('simple_classifier')
class Net(Model):
    
    def __init__(self, 
                 vocab: Vocabulary, 
                 embedder: TextFieldEmbedder, 
                 encoder: Seq2VecEncoder) -> None:
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.linear = torch.nn.Linear(in_features = encoder.get_output_dim(), out_features = num_labels)

        self.accuracy = CategoricalAccuracy()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, 
                text: Dict[str, torch.Tensor], 
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        print('\nIn shape:  {}'.format(self.embedder.get_input_dim()))
        embedded_text = self.embedder(text)
        print('\nOut shape: {}'.format(embedded_text.size()))
        # Shape: (batch_size, num_tokens)
        mask = get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.linear(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output = {'logits':logits, 'probs': probs}
        
        if label is not None:
            self.accuracy(logits, label)
            output['loss'] = self.loss(logits, label)
        return output
        
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'accuracy': self.accuracy.get_metric(reset)}

from typing import Iterator, List, Dict

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer

@DatasetReader.register('tsv-reader')
class TSVDatasetReader(DatasetReader):
    
    def __init__(self, 
                 sentence_tokenizer: Tokenizer = None,
                 verb_tokenizer: Tokenizer = None,
                 sentence_token_indexers: Dict[str, TokenIndexer] = None,
                 verb_token_indexers: Dict[str, TokenIndexer] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.sentence_tokenizer = sentence_tokenizer or WhitespaceTokenizer()
        self.sentence_token_indexers = sentence_token_indexers or {'tokens': SingleIdTokenIndexer()}

        self.verb_tokenizer = verb_tokenizer or WhitespaceTokenizer()
        self.verb_token_indexers = verb_token_indexers or {'tokens': SingleIdTokenIndexer()}

    def text_to_instance(self, sentence: str, verb: str, label: str = None) -> Instance:
        fields = dict()
        
        sentence_tokens = self.sentence_tokenizer.tokenize(sentence)
        sentence_text_field = TextField(sentence_tokens, self.sentence_token_indexers)
        fields['sentence'] = sentence_text_field

        verb_tokens = self.verb_tokenizer.tokenize(verb)
        verb_text_field = TextField(verb_tokens, self.verb_token_indexers)
        fields['verb'] = verb_text_field
        
        if label:
            fields['label'] = LabelField(label)
            
        return Instance(fields)
    
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, 'r') as lines:
            for line in lines:
                sentence, verb, label = line.lower().replace('\n', '').split('\t')
                yield self.text_to_instance(sentence, verb, label)

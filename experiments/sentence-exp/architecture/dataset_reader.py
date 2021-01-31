from typing import Iterator, List, Dict

from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer

@DatasetReader.register('tsv-reader')
class TSVDatasetReader(DatasetReader):
    
    def __init__(self, 
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    def text_to_instance(self, sent: str, label: str = None) -> Instance:
        tokens = self.tokenizer.tokenize(sent)
        text_field = TextField(tokens, self.token_indexers)
        fields = {'text': text_field}

        if label:
            fields['label'] = LabelField(label)
            
        return Instance(fields)
    
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path, 'r') as lines:
            for line in lines:
                sent, _, label = line.replace('\n', '').split('\t')
                yield self.text_to_instance(sent, label)

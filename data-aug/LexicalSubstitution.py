from nltk.corpus import wordnet as wn
import random

class LexicalSubstitution(object):

    @staticmethod
    def synonym_substitution(sentence: str, prob: float = 0.2) -> str:
        """
        Performs synonym substitution on a sentece.

        There's a probability of prob of replacing each word in the sentence
        with its synonym.

            Parameters:
                sentece (str): The original sentence to modify
                prob (float): The probability of replacing a word in the range
                [0, 1). Its default value is 0.2.
            
            Returns:
                new_sentence (str): The modified sentence
                change (boolean): True if there was any replacement performed.
        """

        if 0 > prob >= 1:
            raise Exception('Prob should be a number in the range [0, 1)')
        words = sentence.split()
        new_words = []

        change = False
        for word in words:
            if random.random() > prob:
                new_words.append(_get_word_synonym(word))
                change = True
            else:
                new_words.append(word)
        return ' '.join(new_words), change

    @staticmethod
    def _get_word_synonym(word: str) -> str:
        """
        Get a synonym for a word.

            Parameters:
                word (str): The word to find the synonym of.
            
            Returns:
                synonym (str): The synonym.
        """

        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                if lemma.name() != word:
                    return lemma.name()

        raise Exception('No synonyms found for {}'.format(word))

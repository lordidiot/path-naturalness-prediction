import numpy as np
from nltk.corpus import wordnet
from nltk.corpus.reader.wordnet import Synset
import spacy

GLOVE_PATH = "../data/vectors/glove.42B.300d.txt"

class SenseScore:
    def __init__(self, vertices: set[str]):
        self.vocab = set(vertices)  # to be added on upon caching word senses
        self.nlp = spacy.load("en_core_web_sm")
        self.word_senses = self._cache_word_senses(vertices)
        print("Word senses loaded with size", len(self.word_senses))
        self.glove_embeddings = self._cache_glove_embeddings(self.vocab)
        print("Glove embeddings loaded with size", len(self.glove_embeddings))

    def _cache_glove_embeddings(self, words: set[str]) -> dict[str, np.ndarray]:
        glove_embeddings: dict[str, np.ndarray] = {}
        with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip().split(' ')
                if line[0] in words:
                    glove_embeddings[line[0]] = np.array(line[1:], dtype=np.float32)
        return glove_embeddings
    
    def _cache_word_senses(self, words: set[str]) -> dict[str, list[set[str]]]:
        # Each sense is represented by a set[str],
        # which is the union of synonym, IsA, HasA, definition
        word_senses: dict[str, list[set[str]]] = {}
        for word in words:
            senses: list[set[str]] = []
            for syn in wordnet.synsets(word):
                synonyms = self._get_synonym_words(syn)
                is_a_words = self._get_is_a_words(syn)
                has_a_words = self._get_has_a_words(syn)
                definition_words = self._get_definition_words(syn)
                sense_words = synonyms.union(is_a_words).union(has_a_words).union(definition_words)
                # remove "phrases", not too sure how to handle phrases (with _)
                # remove non-words
                # lower the word because glove stores proper nouns in lower case
                sense_words = set([sense_word.lower() for sense_word in sense_words
                                   if sense_word.isalnum()])
                # remove the word itself
                if word in sense_words:
                    sense_words.remove(word)
                # add the sense words to vocab
                self.vocab = self.vocab.union(sense_words)
                senses.append(sense_words)
            word_senses[word] = senses
        return word_senses
    
    def _get_synonym_words(self, syn: Synset) -> set[str]:
        return set([lemma.name() for lemma in syn.lemmas()])
    
    def _get_word(self, syn: Synset) -> str:
        return syn.lemmas()[0].name()
    
    def _get_is_a_words(self, syn: Synset) -> set[str]:
        hyponyms = [self._get_word(s) for s in syn.hyponyms()]
        hypernyms = [self._get_word(s) for s in syn.hypernyms()]
        return set(hyponyms + hypernyms)
    
    def _get_has_a_words(self, syn: Synset) -> set[str]:
        parts = [self._get_word(s) for s in syn.part_meronyms()]
        substances = [self._get_word(s) for s in syn.substance_meronyms()]
        members = [self._get_word(s) for s in syn.member_meronyms()]
        return set(parts + substances + members)
    
    def _get_definition_words(self, syn: Synset) -> set[str]:
        doc = self.nlp(syn.definition())
        return set(token.lemma_ for token in doc if not token.is_stop)

    def _similarity(self, word1: str, word2: str) -> float:
        return (self.glove_embeddings[word1] @ self.glove_embeddings[word2]).item() \
            / (np.linalg.norm(self.glove_embeddings[word1]) * np.linalg.norm(self.glove_embeddings[word2]))
    
    def sense_similarity(self, word: str, sense: set[str]) -> float:
        # some sense words are not in glove
        similarities = [self._similarity(word, sense_word) for sense_word in sense
                        if sense_word in self.glove_embeddings]
        similarities.sort(reverse=True)
        similarities = similarities[:10]
        return sum(similarities) / len(similarities)

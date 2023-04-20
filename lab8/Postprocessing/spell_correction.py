""" Simple spell correction based on statistical methods """

import re
from collections import Counter
import sys


class SpellCorrection:
    def __init__(self, text_correction_file: str, charlist: str) -> None:
        """ Simple spell correction based on statistical methods """
        self.all_words = Counter(self.words(open(text_correction_file).read()))
        self.charlist = charlist
        
    def words(self, text: str) -> list:
        """ Preprocess words from file """
        return re.findall(r'\w+', text.lower())

    def P(self, word: str) -> float: 
        """ Probability of `word`."""
        N = sum(self.all_words.values())
        return self.all_words[word] / N

    def candidates(self, word: str) -> list: 
        """ Generate possible spelling corrections for word."""
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

    def known(self, words: set) -> set: 
        """ The subset of `words` that appear in the dictionary of WORDS."""
        return set(w for w in words if w in self.all_words)

    def edits1(self, word: str) -> set:
        """ All edits that are one edit away from `word`."""
        letters    = self.charlist
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word: str) -> set: 
        """ All edits that are two edits away from `word`. """
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    def correction(self, word: str) -> str: 
        """ Most probable spelling correction for word."""
        isUpper = word[0].isupper()
        word = word.lower()
        corrected_word =  max(self.candidates(word), key=self.P)
        if(isUpper):
            corrected_word = corrected_word[0].upper() + corrected_word[1:]
        return corrected_word
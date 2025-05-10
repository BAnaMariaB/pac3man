import random
import glob
import sys

"""
Markov Babbler

After being trained on text from various authors, can
'babble', or generate random walks, and produce text that
vaguely sounds like the author.
"""

class Babbler:
    def __init__(self, n, seed=None):
        """
        n is the length of an n-gram for state.
        seed is the seed for a random number generation. If none given use the default.
        """
        self.n = n
        if seed is not None:
            random.seed(seed)
        # transitions: map n-gram tuples to list of successor words
        self._transitions = {}
        # n-grams that start sentences
        self._starters = []
        # n-grams that end sentences
        self._stoppers = []
    
    def add_sentence(self, sentence):
        """
        Process the given sentence.
        The sentence is a string separated by spaces. Break it into
        words using split(). Convert each word to lowercase using lower().
        Then start processing n-grams and updating your states.
        Remember to track starters (i.e. n-grams that begin sentences),
        stoppers (i.e. n-grams that end a sentence), and that
        any n-grams that stop a sentence should be followed by the
        special symbol 'EOL' in the state transition table.
        'EOL' is short for 'end of line'.
        """
        if not sentence:
            return
        # normalize and split
        words = [w.lower() for w in sentence.split()]
        if len(words) < self.n:
            return

        # record starter n-gram
        starter = tuple(words[:self.n])
        self._starters.append(starter)

        # record transitions for all windows
        for i in range(len(words) - self.n):
            ngram = tuple(words[i:i + self.n])
            successor = words[i + self.n]
            self._transitions.setdefault(ngram, []).append(successor)

        # record last n-gram -> EOL
        last = tuple(words[-self.n:])
        self._transitions.setdefault(last, []).append('EOL')
        self._stoppers.append(last)

    def add_file(self, filename):
        """
        This method done for you. It just calls your add_sentence() method
        for each line of an input file. We are assuming that the input data
        has already been pre-processed so that each sentence is on a separate line.
        """
        for line in [line.rstrip().lower() for line in open(filename, errors='ignore').readlines()]:
            self.add_sentence(line)
    
    def get_starters(self):
        """
        Return a list of all of the n-grams that start any sentence we've seen.
        The resulting list may contain duplicates, because one n-gram may start
        multiple sentences.
        """
        #return list(self._starters)
        return[" ".join(ngrams)for ngrams in self._starters]
    
    def get_stoppers(self):
        """
        Return a list of all the n-grams that stop any sentence we've seen.
        The resulting value may contain duplicates, because one n-gram may stop
        multiple sentences.
        """
        #return list(self._stoppers)
        return[" ".join(ngrams)for ngrams in self._stoppers]

    def get_successors(self, ngram):
        """
        Return a list of words that may follow a given n-gram.
        The resulting list may contain duplicates.
        If the given state never occurs, return an empty list.
        """
        return list(self._transitions.get(ngram, []))
    
    def get_all_ngrams(self):
        """
        Return all the possible n-grams, or n-word sequences, that we have seen
        across all sentences.
        """
        return [" ".join(ngrams) for ngrams in self._transitions.keys()]


    def _parse_ngram(self, ngram):
        """
        Internal: accept a string 'w1 w2 ... wn' or tuple, return tuple of words.
        """
        if isinstance(ngram, str):
            parts = ngram.split()
            return tuple(parts)
        return ngram

    def get_successors(self, ngram):
        """
        Return a list of words that may follow a given n-gram (string or tuple),
        with duplicates to reflect frequency. If unseen, returns [].
        """
        key = self._parse_ngram(ngram)
        return list(self._transitions.get(key, []))
    


    def has_successor(self, ngram):
        """
        Return True if the given ngram has at least one possible successor
        word, and False if it does not.
        """
        #return ngram in self._transitions and bool(self._transitions[ngram])
        key = self._parse_ngram(ngram)
        return bool(self._transitions.get(key))

    
    def get_random_successor(self, ngram):
        """
        Given an n-gram, randomly pick from the possible words
        that could follow that n-gram, weighted by frequency.
        """
        succs = self._transitions.get(ngram)
        if not succs:
            return None
        return random.choice(succs)
    
    def babble(self):
        """
        Generate a random sentence using:
        1: Pick a starter ngram.
        2: Choose a successor word based on the current ngram.
        3: If the successor word is 'EOL', return the sentence.
        4: Otherwise, append and slide the window.
        5: Repeat until 'EOL'.
        """
        if not self._starters:
            return ""

        current = random.choice(self._starters)
        sentence = list(current)

        while True:
            nxt = self.get_random_successor(current)
            if nxt is None or nxt == 'EOL':
                break
            sentence.append(nxt)
            current = tuple(sentence[-self.n:])

        return " ".join(sentence)
                

def main(n=3, filename='tests/test1.txt', num_sentences=5):
    """
    Simple test driver.
    """
    print(filename)
    babbler = Babbler(n)
    babbler.add_file(filename)
        
    print(f'num starters {len(babbler.get_starters())}')
    print(f'num ngrams {len(babbler.get_all_ngrams())}')
    print(f'num stoppers {len(babbler.get_stoppers())}')
    for _ in range(num_sentences):
        print(babbler.babble())


if __name__ == '__main__':
    # remove the first parameter, which should be babbler.py, the name of the script
    sys.argv.pop(0)
    n = 3
    filename = 'tests/test1.txt'
    num_sentences = 5
    if len(sys.argv) > 0:
        n = int(sys.argv.pop(0))
    if len(sys.argv) > 0:
        filename = sys.argv.pop(0)
    if len(sys.argv) > 0:
        num_sentences = int(sys.argv.pop(0))
    main(n, filename, num_sentences)

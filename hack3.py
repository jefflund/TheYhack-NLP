import collections
import math

class MarkovModel(object):
    """A simple n-gram language model"""

    def __init__(self, data, order=2, prefix='^', suffix='$'):
        self.n = order - 1
        self.prefix = prefix * self.n
        self.suffix = suffix

        self.table = collections.Counter()
        self.margin = collections.Counter()
        for sequence in data:
            for event, context in self.extract_ngrams(sequence):
                self.table[event, context] += 1
                self.margin[context] += 1

    def extract_ngrams(self, sequence):
        """Generates each n-gram from a sequence.

        As on the whiteboard, the sequence is padded with special symbols so
        every event has the proper context.
        """
        sequence = self.prefix + sequence + self.suffix
        for i, event in enumerate(sequence[self.n:], self.n):
            yield event, sequence[i-self.n: i]

    def cond_prob(self, event, context):
        """Computes the conditional probability of an event given context.

        On the whiteboard, this was p(w_i|w_{i-1},...,w_{i-m}),
        where w_i is the event, and w_{i-1},...,w_{i-m} is the context.
        """
        return self.table[event, context] / self.margin[context]

    def prob(self, sequence):
        """Computes the probability of a sequence.

        On the whiteboard, this was p(w),
        where w is the sequence of events w_1,...,w_n.
        """
        prob = 1
        for event, context in self.extract_ngrams(sequence):
            prob *= self.cond_prob(event, context)
        return prob

    def gen_cond(self, context):
        pass

    def gen(self):
        pass

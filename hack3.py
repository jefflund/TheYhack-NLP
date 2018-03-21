import collections
import math
import random


def sample_categorical(distribution):
    """Samples a value from a categorical distribution.

    The distribution should be given as a dictionary mapping events to their
    respetive probabilities. Those probabilities should sum to 1.
    """
    sample = random.random()
    for event, prob in distribution.items():
        if sample < prob:
            return event
        sample -= prob
    raise ValueError('sum of distribution less than one')


class MarkovModel(object):
    """A simple n-gram language model"""

    def __init__(self, data, order=2, prefix='^', suffix='$'):
        self.n = order - 1
        self.prefix = prefix * self.n
        self.suffix = suffix

        # TODO Add something which tracks all event types
        # Hints
        #   Use a set, and just add each event as you see them

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

    def cond_gen(self, context):
        """Samples the next state from a given context.

        On the whiteboard, picking a value from p(w_i|w_{i_1},...,w_{i-m}).
        """
        # TODO Implement this!
        # Hints
        #   Use sample_categorical to sample from the conditional distribution
        #   To build the distribution, use cond_prob for each possible event
        #   To determine each event type, see the todo in the init

    def gen(self):
        """Samples a sequence by running the Markov process."""
        # TODO Implement this!
        # Hints
        #   Generate each individual event in the sequence using cond_gen
        #   Each call to cond_gen uses context which is the last n events
        #   The first context should just be the prefix
        #   Stop generating events when the suffix (stop symbol) is generated
        #   Strip the prefix and suffix from the generated sequence


# Hopefully this prints out some cool names!
data = open('data/pokemon.txt').read().split('\n')
model = MarkovModel(data, order=3)
for _ in range(10):
    print(model.gen())

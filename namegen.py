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
    """A simple n-gram language model."""

    def __init__(self, data, order=2, prefix='^', suffix='$', prior=1e-10):
        assert order > 0
        self.n = order - 1
        self.prefix = prefix * self.n
        self.suffix = suffix
        self.prior = 1e-10

        self.alphabet = set()
        self.table = collections.Counter()
        self.margin = collections.Counter()
        for sequence in data:
            for event, context in self.extract_ngrams(sequence):
                self.alphabet.add(event)
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
        return (self.table[event, context] + self.prior) / (self.margin[context] + self.prior * len(self.alphabet))

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

        On the whiteboard, that is to say we sample from the distribution
        p(w_i|w_{i_1},...,w_{i-m}).
        """
        dist = {}
        for event in self.alphabet:
            dist[event] = self.cond_prob(event, context)
        return sample_categorical(dist)

    def gen(self):
        """Samples a sequence by running the Markov process.

        On the whiteboard, that is to say we sample from the distribution p(w).
        """
        sequence = self.prefix
        while not sequence.endswith(self.suffix):
            sequence += self.cond_gen(sequence[len(sequence)-self.n:])
        return sequence[len(self.prefix): -len(self.suffix)]


class InterpolatedModel(MarkovModel):
    """A MarkovModel with Jelinek-Mercer smoothing."""

    def __init__(self, data, lambda_, order=2, prefix='^', suffix='$'):
        assert order > 1
        super().__init__(data, order, prefix, suffix)
        self.lambda_ = lambda_
        if order == 2:
            self.lower = MarkovModel(data, order-1, prefix, suffix)
        else:
            self.lower = InterpolatedModel(data, lambda_, order-1, prefix, suffix)

    def cond_prob(self, event, context):
        """Computes the conditional probability of an even given context.

        The conditonal probability is computed using Jelinek-Mercer smoothing
        with a lower order model.
        """
        high = super().cond_prob(event, context)
        low = self.lower.cond_prob(event, context[1:])
        return high * self.lambda_ + low * (1-self.lambda_)


# Hopefully this prints out some cool names!
data = open('data/pokemon.txt').read().split('\n')
model = InterpolatedModel(data, .9, order=5)
for _ in range(10):
    print(model.gen())

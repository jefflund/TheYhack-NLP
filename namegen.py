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

    def __init__(self, data, order=2, prefix='^', suffix='$', prior=0):
        assert order > 0
        self.n = order - 1
        self.prefix = prefix * self.n
        self.suffix = suffix
        self.prior = prior

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
        count = self.table[event, context] + self.prior
        norm = self.margin[context] + (self.prior * len(self.alphabet))
        return count / norm

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

    def evaluate(self, test_data):
        """Computes the log-likelihood of the data for a model."""
        log_like = 0
        for sequence in test_data:
            for event, context in self.extract_ngrams(sequence):
                log_like += math.log(self.cond_prob(event, context))
        return log_like


class KatzMarkovModel(MarkovModel):
    """KatzMarkovModel is a MarkovModel which employs Katz's backoff"""

    def __init__(self, data, order=2, k=0, prefix='^', suffix='$', prior=0):
        assert order > 1
        super().__init__(data, order, prefix, suffix, prior)
        self.k = k
        if order == 2:
            self.backoff = MarkovModel(data, 1, prefix, suffix, prior)
        else:
            self.backoff = KatzMarkovModel(data, order-1, k, prefix, suffix, prior)

    def cond_prob(self, event, context):
        """Computes the conditional probability of an even given context.

        The conditonal probability is computed using Katz's backoff and a
        lower-order model. However, we do not use Good-Turing frequency
        estimation as a discount.
        """
        if self.margin[context] > self.k:
            return super().cond_prob(event, context)
        else:
            return self.backoff.cond_prob(event, context[1:])


# Hopefully this prints out some cool names!
data = open('data/pokemon.txt').read().split('\n')
train_data, test_data = data[:700], data[700:]
model = KatzMarkovModel(train_data, order=4, prior=.001)
print(model.evaluate(test_data))
for _ in range(10):
    print(model.gen())

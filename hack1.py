import collections

class MarkovModel(object):
    """A simple n-gram language model"""

    def __init__(self, data, order=2, prefix='^', suffix='$'):
        self.n = order - 1
        self.prefix = prefix * self.n
        self.suffix = suffix

        # TODO Save counts needed for cond_prob from data
        # Hints:
        #   Each sequence in the data can be split with extract_ngrams
        #   You need both the transition counts and the context counts
        #   Consider using collections.Counter for zero defaults
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
        # TODO Implement this!
        # Hints:
        #   You'll need the counts saved in the model init
        #   Make sure you handle the case of zero counts
        #   Don't overthink it! This could be a one liner :)

    def prob(self, sequence):
        """Computes the probability of a sequence.

        On the whiteboard, this was p(w),
        where w is the sequence of events w_1,...,w_n.
        """
        prob = 1
        for event, context in self.extract_ngrams(sequence):
            prob *= self.cond_prob(event, context)
        return prob


def test_model():
    # This is the data from the whiteboard. Your counts should match the board.
    data = [
        'rssrr',
        'rrrsss',
        'rsr',
    ]
    model = MarkovModel(data)
    assert model.cond_prob('s', '^') == 0
    assert model.cond_prob('r', '^') == 3/3
    assert model.cond_prob('s', 'r') == 3/8
    assert model.cond_prob('$', 's') == 1/6
    assert model.prob('rs') == .0625

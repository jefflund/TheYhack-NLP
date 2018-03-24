class MarkovModel(object):
    """A simple n-gram language model"""

    def __init__(self, order=2, prefix='^', suffix='$'):
        self.n = order - 1
        self.prefix = prefix * self.n
        self.suffix = suffix

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
        # This is clearly wrong and nonsensical. It is just here as a stub.
        # Don't worry we'll fix this next! For now just do implement prob.
        return {'rs':.2,'rr':.1,'sr':.25,'ss':.5}.get(event+context, .125)

    def prob(self, sequence):
        """Computes the probability of a sequence.

        On the whiteboard, this was p(w),
        where w is the sequence of events w_1,...,w_n.
        """
        # TODO Implement this!
        # Hints:
        #   You'll probably want to use extract_ngrams to split the sequence
        #   extract_ngrams yields a sequence of (event, context) tuples
        #   For each n-gram, you'll want to use cond_prob in some way


def test_model():
    model = MarkovModel()
    assert round(model.prob('rrsssr'), 10) == 1.95313e-05
    assert round(model.prob('rsrrrs'), 10) == 1.9531e-06
    assert round(model.prob('rsrsss'), 10) == 4.88281e-05

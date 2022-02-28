import numpy as np


class ProbDist:
    """A discrete probability distribution. You name the random variable
    in the constructor, then assign and query probability of values.
    Values must be of type string.
    >>> P = ProbDist('Flip'); P['H'], P['T'] = 0.25, 0.75; P['H']
    0.25
    >>> P = ProbDist('X', lo=125, med=375, hi=500})
    >>> P = ProbDist('X', **{'lo': 125, 'med': 375, 'hi': 500})
    >>> P['lo'], P['med'], P['hi']
    (0.125, 0.375, 0.5)
    """

    def __init__(self, var_name='?', **freq_dict):
        """freq_dict are keyword arguments for value, probability pairs."""
        self.prob = {}
        self.var_name = var_name

        if freq_dict:
            for (v, p) in freq_dict.items():
                self[v] = p

        self.normalize()

    @property
    def variable_values(self):
        return tuple(self.prob.keys())

    def __getitem__(self, val):
        """Given a value, return P(value)."""
        try:
            return self.prob[val]
        except KeyError:
            return 0

    def __setitem__(self, val: str, p):
        """Set P(val) = p."""
        if not isinstance(val, str):
            raise ValueError(f"Value must be of type str. <{val}> is of {type(val)}.")
        self.prob[val] = p

    def normalize(self):
        """Make sure the probabilities of all values sum to 1.
        Returns the normalized distribution.
        Raises a ZeroDivisionError if the sum of the values is 0."""
        total = sum(self.prob.values())
        if not np.isclose(total, 1.0):
            for val in self.prob:
                self.prob[val] /= total
        return self

    def show_approx(self, numfmt='{:.3g}'):
        """Show the probabilities rounded."""
        return ', '.join([('{}: ' + numfmt).format(v, p) for (v, p) in self.prob.items()])

    def __repr__(self):
        return f"P({self.var_name})=<{self.show_approx()}>"

from functools import reduce

import pandas as pd

from probinf.distributions import ProbDist
from probinf.utils import all_events, event_values, extend


def query_check(X, e, bn):
    """Checks if query only contains valid variables and values"""

    if X in e:
        raise ValueError(f"Query variable <{X}> must be distinct from evidence.")

    variables = bn.variables

    if X not in variables:
        raise ValueError(f"Query variable <{X}> not in net.")

    for variable, value in e.items():
        if variable not in variables:
            raise ValueError(f"Evidence variable <{variable}> not in net.")
        if value not in bn.variable_values(variable):
            raise ValueError(f"Value <{value}> of variable <{variable}> not in cpt.")


# ______________________________________________________________________________

def enumerate_all(variables, e, bn):
    """Return the sum of those entries in P(variables | e{others})
    consistent with e, where P is the joint distribution represented
    by bn, and e{others} means e restricted to bn's other variables
    (the ones other than variables). Parents must precede children in variables."""
    if not variables:
        return 1.0
    Y, rest = variables[0], variables[1:]
    Ynode = bn.get_node(Y)
    if Y in e:
        return Ynode.p(e[Y], e) * enumerate_all(rest, e, bn)
    else:
        return sum(Ynode.p(y, e) * enumerate_all(rest, extend(e, Y, y), bn)
                   for y in bn.variable_values(Y))


def enumeration_ask(X, e, bn):
    """
    Return the conditional probability distribution of variable X
    given evidence e, from BayesNet bn.
    >>> enumeration_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary
    ...  ).show_approx()
    'False: 0.716, True: 0.284'"""

    query_check(X, e, bn)

    dist_name = f"{X}|{', '.join([f'{var}={value}' for var, value in e.items()])}"

    Q = ProbDist(dist_name)
    for xi in bn.variable_values(X):
        Q[xi] = enumerate_all(bn.variables, extend(e, X, xi), bn)
    return Q.normalize()


# ______________________________________________________________________________

class Factor:
    """A factor in a joint distribution."""

    def __init__(self, variables, cpt):
        self.variables = variables
        self.cpt = cpt

    def pointwise_product(self, other, bn):
        """Multiply two factors, combining their variables."""
        variables = list(set(self.variables) | set(other.variables))
        cpt = {event_values(event, variables): self.p(event) * other.p(event) for event in
               all_events(variables, bn, {})}
        return Factor(variables, cpt)

    def sum_out(self, var, bn):
        """Make a factor eliminating var by summing over its values."""
        variables = [X for X in self.variables if X != var]
        cpt = {event_values(event, variables): sum(self.p(extend(event, var, val))
                                                   for val in bn.variable_values(var))
               for event in all_events(variables, bn, {})}
        return Factor(variables, cpt)

    def normalize(self):
        """Return my probabilities; must be down to one variable."""

        if len(self.variables) != 1:
            raise ValueError(
                f"Factor must only contain one variable to normailze. Amount of variables {len(self.variables)}.")

        return ProbDist(self.variables[0], **{k: v for ((k,), v) in self.cpt.items()})

    def p(self, e):
        """Look up my value tabulated for e."""
        return self.cpt[event_values(e, self.variables)]

    def as_table(self):

        columns = self.variables + [f"f({self.variables})"]

        data = []
        for value_combination, prob in self.cpt.items():
            row = value_combination + (prob,)
            data.append(row)
        df = pd.DataFrame(data=data, columns=columns)
        df.set_index(self.variables, inplace=True)

        return df

    @staticmethod
    def make_factor(var, evidence, bn):
        """Return the factor for var in bn's joint distribution given evidence.
        That is, bn's full joint distribution, projected to accord with evidence,
        is the pointwise product of these factors for bn's variables."""
        node = bn.get_node(var)
        variables = [X for X in [var] + node.parents if X not in evidence]
        cpt = {event_values(event, variables): node.p(event[var], event)
               for event in all_events(variables, bn, evidence)}
        return Factor(variables, cpt)


def pointwise_product(factors, bn):
    """Pointwise product between mor than two factors"""
    return reduce(lambda f, g: f.pointwise_product(g, bn), factors)


def sum_out(var, factors, bn):
    """Eliminate var from all factors by summing over its values."""
    result, var_factors = [], []
    for factor in factors:
        if var in factor.variables:
            var_factors.append(factor)
        else:
            result.append(factor)
    result.append(pointwise_product(var_factors, bn).sum_out(var, bn))
    return result


def is_hidden(var, X, e):
    """Is var a hidden variable when querying P(X|e)?"""
    return var != X and var not in e


def elimination_ask(X, e, bn):
    """
    Compute bn's P(X|e) by variable elimination.
    >>> elimination_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary
    ...  ).show_approx()
    'False: 0.716, True: 0.284'"""

    query_check(X, e, bn)

    factors = []
    for var in reversed(bn.variables):
        factors.append(Factor.make_factor(var, e, bn))
        if is_hidden(var, X, e):
            factors = sum_out(var, factors, bn)

    dist = pointwise_product(factors, bn).normalize()
    dist.var_name = f"{X}|{', '.join([f'{var}={value}' for var, value in e.items()])}"

    return dist

import itertools

import pandas as pd

from probinf import utils
from probinf.distributions import ProbDist
from probinf.utils import F, T


class BayesNode:
    """A node in a BayesNet with conditional probability distribution P(X | parents)"""

    def __init__(self, variable, parents, cpt):
        """X is a variable name, and parents a sequence of variable
        names or a space-separated string. cpt_dict, a dictionary representing the conditional
        probability table, which is internally transformed to a ConditionalProbabilityTable.
        cpt_dict takes one of these forms::

        * A number, the unconditional probability P(X=true). You can
          use this form when there are no parents.

        * A dict {v: p, ...}, the conditional probability distribution
          P(X=true | parent=v) = p. When there's just one parent.

        * A dict {(v1, v2, ...): ProbDist(...), ...}, the distribution P(X=true |
          parent1=v1, parent2=v2, ...) = ProbDist(...). Each key must have as many
          values as there are parents. You can use this form always;
          the first two are just conveniences.

        In case of a boolean variable the probability being false is left implicit,
        in all other cases there must be a ProbDist() specified. All distribution values must be strings.
        For boolean variables use varaibles T,F from utils module.

        >>> X = BayesNode('X', '', 0.2)
        >>> X = BayesNode('X', '', ProbDist(lo=125, med=375, hi=500))

        >>> Y = BayesNode('Y', 'P', {T: 0.2, F: 0.7})
        >>> Y = BayesNode('Y', 'P',
        >>>               {T: ProbDist(lo=125, med=375, hi=500), F: ProbDist(lo=125, med=375, hi=500)})

        >>> Z = BayesNode('Z', 'P', {(T,): 0.2, (F,): 0.3})
        >>> Z = BayesNode('Z', 'P Q', {(T, T): 0.2, (T, F): 0.3, ... })
        >>> Z = BayesNode('Z', 'P Q', {('lo', T): ProbDist(lo=125, med=375, hi=500), ('med',T): ProbDist(...), ...})
        """

        if isinstance(parents, str):
            parents = parents.split()

        self.variable = variable
        self.parents = parents

        # convenience no parents, 0-tuple
        if isinstance(cpt, (float, int, ProbDist)):
            cpt = {(): cpt}
        elif isinstance(cpt, dict):
            # convenience one parent, 1-tuple
            if not isinstance(list(cpt.keys())[0], tuple):
                cpt = {(v,): p for v, p in cpt.items()}
        else:
            raise CPTTypeError(f"CPT has to be of type int, float, dict or ProbDist. "
                               f"Node: <{self.variable}>; Type:{type(cpt)}.")

        for parent_values, prob_dist in cpt.items():

            parent_value_str = ', '.join([f"{var}={value}" for var, value in zip(parents, parent_values)])
            dist_name = f"{variable}|{parent_value_str}"

            # Convenience only true value provided
            if isinstance(prob_dist, (float, int)):
                cpt[parent_values] = ProbDist(dist_name, **{T: prob_dist, F: 1 - prob_dist})
            # Normal case
            elif isinstance(prob_dist, ProbDist):
                prob_dist.var_name = dist_name
            else:
                raise CPTTypeError(
                    f"The distribution has to be of type int, float or ProbDist. "
                    f"Node: <{self.variable}>; Type:{type(prob_dist)}.")

        self.cpt = cpt
        self.parent_check()
        self.value_check()

    @property
    def variable_values(self):
        prob_dist = list(self.cpt.values())[0]
        return prob_dist.variable_values

    def parent_check(self):
        """Checks if amount of specified parents equals the amount of values inside each cpt key."""
        for row, parent_values in enumerate(self.cpt.keys()):
            if len(parent_values) != len(self.parents):
                raise CPTParentError(f"Amount of specified parents <{self.parents}> doesn't equal the amount of"
                                     f" values inside the cpt key <{parent_values}> "
                                     f"in row <{row+1}> of node <{self.variable}>.")

    def value_check(self):
        """Checks if the values of the distribution of variable X are the same in all table rows"""
        prob_dist_values = set({})
        for prob_dist in self.cpt.values():
            prob_dist_values.add(prob_dist.variable_values)
        if len(prob_dist_values) > 1:
            raise CPTValueError(
                f"The values of the distribution of variable <{self.variable}> must be the same in all table rows."
                f" Specified values: {prob_dist_values}")

    def as_table(self):
        """Returns the cpt in form of a pandas DataFrame"""
        columns = self.parents + list(self.variable_values)

        data = []
        for parent_assignments, prob_dist in self.cpt.items():
            row = parent_assignments + tuple(prob_dist.prob.values())
            data.append(row)
        df = pd.DataFrame(data=data, columns=columns)

        if self.parents:
            df.set_index(self.parents, inplace=True)
        return df

    def p(self, value, event):
        """Return the conditional probability
        P(X=value | parents=parent_values), where parent_values
        are the values of parents in event. (event must assign each
        parent a value.)
        >>> bn = BayesNode('X', 'Burglary', {T: 0.2, F: 0.625})
        >>> bn.p(F, {'Burglary': F, 'Earthquake': T})
        0.375"""
        parent_values = utils.event_values(event, self.parents)
        p = self.cpt[parent_values][value]
        return p

    def __repr__(self):
        return repr(f"Name: {self.variable}; Parents: {', '.join(self.parents)}")


class BayesNet:
    """Bayesian network containing variable nodes with discrete probability distributions."""

    def __init__(self, node_specs=None):
        """Nodes must be ordered with parents before children.
        node_specs is a list of tuples, each tuple contains 3 elements,
        specifying the parameters of the BayesNode constructor (X, parents, cpt_dict)."""
        self.nodes = []
        node_specs = node_specs or []
        for node_spec in node_specs:
            self.add(node_spec)
        self.cpt_check()

    @property
    def variables(self):
        return [node.variable for node in self.nodes]

    def add(self, node):
        """Add a node to the net. Its parents must already be in the
        net, and its variable must not. node Parameter can be either a BayesNode or
        the parameters of the BayesNode constructor (X, parents, cpt_dict)."""
        if isinstance(node, tuple):
            node = BayesNode(*node)
        elif not isinstance(node, BayesNode):
            raise ValueError(f"Node must be of type tuple or BayesNode not {type(node)}")

        if node.variable in self.variables:
            raise ValueError(f"Node <{node.variable}> already in net.")
        if not all((parent in self.variables) for parent in node.parents):
            raise ValueError(f"Not all parents in net for Node <{node.variable}>.")

        self.nodes.append(node)

    def cpt_check(self):
        """Checks if all possible value combinations of parents are enumerated correctly in cpt of node

           Causes of error could be:
           * Not all possible combinations are specified
           * Invalid parent values are specified (Values that are not part of the distribution of the parent)"""

        for node in self.nodes:
            parent_values = [self.get_node(parent).variable_values for parent in node.parents]
            if set(node.cpt.keys()) != set(itertools.product(*parent_values)):
                raise CPTParentError(
                    f"Value combinations of parents are not enumerated properly in cpt of node <{node.variable}>.\n"
                    f" Possible reasons:\n"
                    f" * Not all possible combinations are specified\n"
                    f" * Invalid parent values are specified"
                    f" (Values that are not part of the distribution of the parent)")

    def get_node(self, var):
        """Return the node for the variable named var.
        >>> burglary.get_node('Burglary').variable"""
        try:
            return self.nodes[self.variables.index(var)]
        except ValueError:
            raise Exception(f"No such variable <{var}> in net.")

    def variable_values(self, var):
        """Return the domain of var."""
        return self.get_node(var).variable_values

    def __repr__(self):
        return 'BayesNet({0!r})'.format(self.nodes)


# ______________________________________________________________________________
# Custom exception classes

class CPTError(Exception):
    """Basic exception class for exceptions related to ConditionalProbabilityTable"""
    pass


class CPTValueError(CPTError):
    pass


class CPTParentError(CPTError):
    pass


class CPTTypeError(CPTError):
    pass

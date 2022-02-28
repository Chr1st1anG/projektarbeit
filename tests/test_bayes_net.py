import pandas as pd
import pytest

from probinf.bayes_net import BayesNet, BayesNode, CPTTypeError, CPTParentError, CPTValueError
from probinf.distributions import ProbDist
from probinf.utils import F, T


def test_bayes_net_init():
    net = BayesNet([
        ('Vaccinated', [], 0.60),  # convenience no parent
        ('Flu', ['Vaccinated'], {T: 0.002, F: 0.02}),  # convenience one parent
        ('Fever', ['Flu'], {T: ProbDist(no=25, mild=25, high=50),  # non binary distribution
                            F: ProbDist(no=97, mild=2, high=1)}),
        ('Headache', 'Flu Vaccinated',  # more than one parent
         {(T, T): 0.95, (T, F): 0.94, (F, T): 0.29, (F, F): 0.001})
    ])

    assert net.get_node('Vaccinated').cpt[()][F] == 0.4
    assert net.variable_values('Vaccinated') == (T, F)
    assert net.get_node('Flu').cpt[(F,)][F] == 0.98
    assert isinstance(net.get_node('Flu').as_table(), pd.DataFrame)
    assert net.get_node('Fever').cpt[(F,)]["mild"] == 0.02
    assert net.get_node('Fever').variable_values == ('no', 'mild', 'high')
    assert net.get_node('Headache').parents == ['Flu', 'Vaccinated']
    assert net.get_node('Headache').p(F, {'Flu': F, 'Vaccinated': T, 'NoParent': T}) == 0.71


def test_bayes_net_add():
    vac = BayesNode('Vaccinated', [], 0.60)
    flu = BayesNode('Flu', ['Vaccinated'], {T: 0.002, F: 0.02})
    fev = BayesNode('Fever', ['Flu'], {T: ProbDist(no=25, mild=25, high=50), F: ProbDist(no=97, mild=2, high=1)})
    hea = BayesNode('Headache', 'Flu Vaccinated', {(T, T): 0.95, (T, F): 0.94, (F, T): 0.29, (F, F): 0.001})

    net = BayesNet()
    net.add(vac)
    net.add(flu)
    net.add(fev)
    net.add(hea)

    assert net.get_node('Vaccinated').cpt[()][F] == 0.4
    assert net.variable_values('Vaccinated') == (T, F)
    assert net.get_node('Flu').cpt[(F,)][F] == 0.98
    assert isinstance(net.get_node('Flu').as_table(), pd.DataFrame)
    assert net.get_node('Fever').cpt[(F,)]["mild"] == 0.02
    assert net.get_node('Fever').variable_values == ('no', 'mild', 'high')
    assert net.get_node('Headache').parents == ['Flu', 'Vaccinated']
    assert net.get_node('Headache').p(F, {'Flu': F, 'Vaccinated': T, 'NoParent': T}) == 0.71


def test_cpt_errors():
    # CPT of type string
    with pytest.raises(CPTTypeError):
        BayesNet([('Vaccinated', [], [0.002, 0.998])])

    # Probability distribution of type list
    with pytest.raises(CPTTypeError):
        BayesNet([('Flu', ['Vaccinated'], {T: [0.002, 0.998], F: [0.002, 0.998]})])

    # Two parents specified but only one parent in cpt (parent_check)
    with pytest.raises(CPTParentError):
        BayesNet([('Vaccinated', [], 0.60),
                  ('Immune', '', 0.5),
                  ('Flu', ['Vaccinated', 'Immune'], {T: 0.002, F: 0.02})])

    # Inconsistency between the distributions inside the cpt. Value <"no" != "wrong"> (value_check)
    with pytest.raises(CPTValueError):
        BayesNet([('Flu', '', 0.5),
                  ('Fever', ['Flu'], {T: ProbDist(no=25, mild=25, high=50),
                                      F: ProbDist(wrong=97, mild=2, high=1)})])

    # Not all possible parent combinations are specified (cpt_check)
    with pytest.raises(CPTParentError):
        BayesNet([('Vaccinated', [], 0.60),
                  ('Immune', '', 0.5),
                  ('Flu', ['Vaccinated', 'Immune'], {(T, T): 0.002, (F, T): 0.02, (F, F): 0.02})])

    # "False" is not part of the distribution of the parent (cpt_check)
    with pytest.raises(CPTParentError):
        BayesNet([('Vaccinated', [], 0.60),
                  ('Immune', '', 0.5),
                  ('Flu', ['Vaccinated', 'Immune'],
                   {(T, T): 0.002, (F, T): 0.02, (T, F): 0.05, (F, F): 0.02, (F, "False"): 0.5})])


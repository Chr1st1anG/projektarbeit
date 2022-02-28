from probinf.bayes_net import BayesNet
from probinf.distributions import ProbDist
from probinf.inference import enumeration_ask, elimination_ask
from probinf.utils import F, T

flu_net = BayesNet([
    ('Vaccinated', [], 0.60),
    ('Flu', ['Vaccinated'], {T: 0.002, F: 0.02}),
    ('Fever', ['Flu'], {T: ProbDist(no=25, mild=25, high=50),
                        F: ProbDist(no=97, mild=2, high=1)}),
    ('Headache', ['Flu'], {T: 0.5, F: 0.03}),
])

burglary = BayesNet([
    ('Burglary', '', 0.001),
    ('Earthquake', '',  0.002),
    ('Alarm', 'Burglary Earthquake',
     {(T, T): 0.95, (T, F): 0.94, (F, T): 0.29, (F, F): 0.001}),
    ('JohnCalls', 'Alarm', {T: 0.90, F: 0.05}),
    ('MaryCalls', 'Alarm', {T: 0.70, F: 0.01})
])


def test_binary_enumeration():
    assert round(enumeration_ask('MaryCalls', dict(Earthquake=T, Alarm=T), burglary)[T], 3) == 0.7
    assert round(enumeration_ask('MaryCalls', dict(Burglary=T), burglary)[T], 3) == 0.659
    assert round(enumeration_ask('Alarm', dict(Burglary=T, Earthquake=T), burglary)[F], 3) == 0.05


def test_binary_elemination():
    assert round(elimination_ask('MaryCalls', dict(Earthquake=T, Alarm=T), burglary)[T], 3) == 0.7
    assert round(elimination_ask('MaryCalls', dict(Burglary=T), burglary)[T], 3) == 0.659
    assert round(elimination_ask('Alarm', dict(Burglary=T, Earthquake=T), burglary)[F], 3) == 0.05


def test_enumeration():
    assert round(enumeration_ask('Fever', {'Headache': T}, flu_net)['mild'], 3) == 0.051
    assert round(enumeration_ask('Flu', {'Headache': T, 'Fever': 'no', 'Vaccinated': T}, flu_net)[T], 3) == 0.009


def test_elemination():
    assert round(elimination_ask('Fever', {'Headache': T}, flu_net)['mild'], 3) == 0.051
    assert round(elimination_ask('Flu', {'Headache': T, 'Fever': 'no', 'Vaccinated': T}, flu_net)[T], 3) == 0.009
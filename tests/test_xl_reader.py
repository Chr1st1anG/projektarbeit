from probinf import xl_reader
from probinf.utils import F


def test_xl_reader():
    bayes_net = xl_reader.get_net_from_xls("data/covid19_for_contract_tracing_paper.xls")

    assert len(bayes_net.variables) == 39
    assert bayes_net.get_node("Amount of contact with virus").variable_values \
           == ("Very High", "High", "Medium", "Low", "None")
    assert bayes_net.get_node("Infected with Covid").cpt[F, "High"][F] == 0.5
    assert bayes_net.get_node("obesity").cpt[()][F] == 0.9

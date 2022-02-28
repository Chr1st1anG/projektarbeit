import xlrd

from probinf.bayes_net import BayesNet, BayesNode
from probinf.distributions import ProbDist
from probinf.utils import T

# ______________________________________________________________________________
# Configuration

TABLE_START_ROW = 2
TABLE_START_COL = 3

MARKER_VALUE = 'VALUE'
MARKER_PARENT = 'PARENT'


# ______________________________________________________________________________
# Functions


def get_net_from_xls(filename=None, file_contents=None):
    """Reads an xls file and creates a BayesNet.
        xlsx is not supported."""
    workbook = xlrd.open_workbook(filename=filename, file_contents=file_contents)
    net = BayesNet()

    for sheet in workbook.sheets():
        X = sheet.name
        parents = _get_parents(sheet)
        values = _get_values(sheet, len(parents))
        cpt = _create_cpt(sheet, X, parents, values)

        node = BayesNode(X, parents, cpt)
        net.add(node)

    net.cpt_check()
    return net


def _get_parents(sheet):
    """Extracts parent names from spreadsheet."""
    return _get_marker_fields(sheet, MARKER_PARENT, TABLE_START_COL)


def _get_values(sheet, len_parents):
    """Extracts the values of the distribution of the variable from spreadsheet."""
    return _get_marker_fields(sheet, MARKER_VALUE, TABLE_START_COL + len_parents)


def _get_marker_fields(sheet, marker, start_idx=0):
    """Extracts values marked with specific marker from spreadsheet.
    Marker is one row above the value to be extracted."""
    i = start_idx
    fields = []
    while True:
        # break if end of the sheet is reached or the marker is not present anymore
        if i >= sheet.ncols or sheet.cell(TABLE_START_ROW, i).value != marker:
            break
        fields.append(str(sheet.cell(TABLE_START_ROW + 1, i).value))
        i += 1

    return fields


def _create_cpt(sheet, X, parents, values):
    """Creates a conditional distribution table from spreadsheet."""
    cpt = {}
    for row in range(TABLE_START_ROW + 2, sheet.nrows):

        # Get parent value assignments
        try:
            parent_config = []
            for col in range(len(parents)):
                parent_value = str(sheet.cell(row, TABLE_START_COL + col).value)
                parent_config.append(parent_value)
            parent_config = tuple(parent_config)
        except ValueError:
            raise ValueError(f"Could not convert parent value to str in cpt of node <{X}>.")

        # Get probabilities
        # Convenience only True probability specified
        if len(values) == 1:
            if values[0] == T:
                try:
                    cpt[parent_config] = float(sheet.cell(row, TABLE_START_COL + len(parents)).value)
                except ValueError:
                    raise ValueError(f"Could not convert probability to float in cpt of node <{X}>.")
            else:
                raise ValueError(
                    f"Value for Variable <{X}> must be 'T' not '{values[0]}', when only one value is specified.")

        # Normal case all values specified
        elif len(values) > 1:
            try:
                probabilities = [float(sheet.cell(row, TABLE_START_COL + len(parents) + col).value)
                                 for col in range(len(values))]
                if abs(sum(probabilities) - 1) > 0.01:
                    print(f"WARNING: CPT probabilities in node <{X}> "
                          f"at row <{row + 1}> don't sum to one. Sum: <{sum(probabilities)}>.")
            except ValueError:
                raise ValueError(f"Could not convert probability to float in cpt of node <{X}>.")
            prob_dist = dict(zip(values, probabilities))
            cpt[parent_config] = ProbDist(**prob_dist)

        else:
            raise ValueError(f"No values provided for Variable <{X}>.")

    return cpt

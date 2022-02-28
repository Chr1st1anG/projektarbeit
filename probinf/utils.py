def event_values(event: dict, variables: list):
    """Return a tuple of the values of variables in event.
    >>> event_values ({'A': 10, 'B': 9, 'C': 8}, ['C', 'A'])
    (8, 10)
    """
    return tuple([event[var] for var in variables])


def all_events(variables: list, bn, e: dict):
    """Yield every way of extending e with values for all specified variables.
    >>> all_events(['Earthquake', 'Alarm'],burglary,{'Burglary':T})
    [{'Burglary': 'T', 'Alarm': 'T', 'Earthquake': 'T'},
     {'Burglary': 'T', 'Alarm': 'T', 'Earthquake': 'F'},
     {'Burglary': 'T', 'Alarm': 'F', 'Earthquake': 'T'},
     {'Burglary': 'T', 'Alarm': 'F', 'Earthquake': 'F'}]
    """
    if not variables:
        yield e
    else:
        X, rest = variables[0], variables[1:]
        for e1 in all_events(rest, bn, e):
            for x in bn.variable_values(X):
                yield extend(e1, X, x)


def extend(s: dict, var, val):
    """Copy dict s and extend it by setting var to val; return copy."""
    return {**s, var: val}


# ______________________________________________________________________________
# Useful Shorthands
T = "T"
F = "F"

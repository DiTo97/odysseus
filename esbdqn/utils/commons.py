import typing as t


def sum_tuple_vals(T: t.List[t.Tuple]) \
                  -> t.Tuple:
    """
    Sum all tuple values independently on axis 1.

    Returns
    -------
    t.Tuple
        Tuple of independent sums.
    """
    return tuple([sum(x) for x in zip(*T)])

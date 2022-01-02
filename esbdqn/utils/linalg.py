import numpy as np
import typing as t


def one_hot(A: t.Any, num_classes: int):
    """
    One-hot encode a value (or a group of values, `t.Iterable`)
    given the number of classes in the codebook.
    """
    return np.squeeze(np.eye(num_classes)[
                      np.asarray(A).reshape(-1)])

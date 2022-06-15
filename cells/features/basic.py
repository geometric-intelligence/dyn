"""Compute basic shape features."""

import numpy as np


def perimeter(xy):
    """Calculate polygon perimeter.

    Parameters
    ----------
    xy : array-like, shape=[n_points, 2]
        Polygon, such that:
        x = xy[:, 0]; y = xy[:, 1]
    """
    xy = xy
    xy1 = np.roll(xy, -1, axis=0)  # shift by -1
    return np.sum(np.sqrt((xy1[:, 0] - xy[:, 0]) ** 2 + (xy1[:, 1] - xy[:, 1]) ** 2))


def area(xy):
    """Calculate polygon area.

    Parameters
    ----------
    xy : array-like, shape=[n_points, 2]
        Polygon, such that:
        x = xy[:, 0]; y = xy[:, 1]
    """
    xy = xy
    n_points = len(xy)
    s = 0.0
    for i in range(n_points):
        j = (i + 1) % n_points
        s += (xy[j, 0] - xy[i, 0]) * (xy[j, 1] + xy[i, 1])
    return -0.5 * s

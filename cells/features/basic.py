"""Compute basic shape features."""

import geomstats.backend as gs


def perimeter(xy):
    """Calculate polygon perimeter.

    Parameters
    ----------
    xy : array-like, shape=[n_points, 2]
        Polygon, such that:
        x = xy[:, 0]; y = xy[:, 1]
    """
    first_point = gs.expand_dims(gs.array(xy[0]), axis=0)
    xy1 = gs.concatenate([xy[1:], first_point], axis=0)
    return gs.sum(gs.sqrt((xy1[:, 0] - xy[:, 0]) ** 2 + (xy1[:, 1] - xy[:, 1]) ** 2))


def area(xy):
    """Calculate polygon area.

    Parameters
    ----------
    xy : array-like, shape=[n_points, 2]
        Polygon, such that:
        x = xy[:, 0]; y = xy[:, 1]
    """
    n_points = len(xy)
    s = 0.0
    for i in range(n_points):
        j = (i + 1) % n_points
        s += (xy[j, 0] - xy[i, 0]) * (xy[j, 1] + xy[i, 1])
    return -0.5 * s

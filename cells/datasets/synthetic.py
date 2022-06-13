"""Utils to load datasets of cells."""

import geomstats.backend as gs


def rectangle(n_points_height, n_points_length, height, length):
    """Return a rectangle as an array of 2D points.

    This function returns a discrete closed curve representing a
    rectangle (height, length) as a array of its 2D points,
    in counter-clockwise order.
    """
    n_points = 2 * n_points_height + 2 * n_points_length - 4
    rectangle = gs.zeros((n_points, 2))

    height_pos_axis = height * gs.linspace(-1, 1, n_points_height)
    height_neg_axis = gs.flip(height_pos_axis)

    length_pos_axis = length * gs.linspace(-1, 1, n_points_length)
    length_neg_axis = gs.flip(length_pos_axis)

    heights = height * gs.ones(n_points_height)
    minus_heights = -heights
    lengths = length * gs.ones(n_points_length)
    minus_lengths = -lengths

    bottom = gs.vstack((length_pos_axis, minus_heights)).T[:-1]
    right = gs.vstack((lengths, height_pos_axis)).T[:-1]
    top = gs.vstack((length_neg_axis, heights)).T[:-1]
    left = gs.vstack((minus_lengths, height_neg_axis)).T[:-1]

    rectangle[: n_points_length - 1] = bottom
    rectangle[n_points_length - 1 : n_points_length + n_points_height - 2] = right
    rectangle[
        n_points_length
        + n_points_height
        - 2 : 2 * n_points_length
        + n_points_height
        - 3
    ] = top
    rectangle[2 * n_points_length + n_points_height - 3 :] = left

    return rectangle


def ellipse(n_points, a, b):
    """Return an ellipse as an array of 2D points.

    This function returns a discrete closed curve representing the ellipse
    of equation: x**2/a**2 + y**2/b**2 =1.
    The discrete closed curve is represented by an array of 2D points
    in counter-clockwise order.

    Notes
    -----
    Area = pi * a * b
    Perimeter = 4 * a * E(e)
    """
    t = gs.linspace(0, 2 * gs.pi, n_points + 1)[:-1]
    x = a * gs.cos(t)
    y = b * gs.sin(t)
    ellipse = gs.vstack((x, y)).T

    return ellipse


def unit_square(n_points_side):
    """Return a square as an array of 2D points.

    This function returns a discrete closed curve representing a
    unit square as a array of its 2D points, in counter-clockwise order.
    """
    return rectangle(n_points_side, n_points_side, 1, 1)


def unit_circle(n_points):
    """Return a circle as an array of 2D points.

    This function returns a discrete closed curve representing the
    unit circle, as an array of 2D points.
    """
    return ellipse(n_points, 1, 1)

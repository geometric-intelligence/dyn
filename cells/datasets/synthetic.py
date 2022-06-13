"""Utils to load datasets of cells."""

import geomstats.backend as gs
import numpy as np
from geomstats.geometry.discrete_curves import R2, DiscreteCurves

CURVES_SPACE = DiscreteCurves(R2)
METRIC = CURVES_SPACE.square_root_velocity_metric


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


def square(n_points_side, side=1):
    """Return a square as an array of 2D points.

    This function returns a discrete closed curve representing a
    unit square as a array of its 2D points, in counter-clockwise order.
    """
    return rectangle(n_points_side, n_points_side, side, side)


def circle(n_points, radius=1):
    """Return a circle as an array of 2D points.

    This function returns a discrete closed curve representing the
    unit circle, as an array of 2D points.
    """
    return ellipse(n_points, 1 / radius, 1 / radius)


def geodesics_square_to_rectangle(n_geodesics=10, n_times=20, n_points=40):
    """Generate a dataset of geodesics that transform squares into rectangles."""
    dim = 2
    n_points_side = n_points_lengh = n_points_heigh = n_points // 4 + 1
    sides = np.random.normal(loc=5, scale=0.2, size=(n_geodesics,))
    heights = np.random.normal(loc=2, scale=0.1, size=(n_geodesics,))
    lengths = np.random.normal(loc=10, scale=0.1, size=(n_geodesics,))

    geodesics = gs.zeros((n_geodesics, n_times, n_points, dim))
    times = gs.arange(0, 1, 1 / n_times)
    for i_geodesic in range(n_geodesics):
        start_square = square(n_points_side=n_points_side, side=sides[i_geodesic])
        end_rect = rectangle(
            n_points_height=n_points_heigh,
            n_points_length=n_points_lengh,
            height=heights[i_geodesic],
            length=lengths[i_geodesic],
        )
        geodesic = METRIC.geodesic(initial_curve=start_square, end_curve=end_rect)
        geodesics[i_geodesic] = geodesic(times)

    return geodesics


def geodesics_circle_to_ellipse(n_geodesics=10, n_times=20, n_points=40):
    """Generate a dataset of geodesics that transform circles into ellipses."""
    dim = 2
    radii = np.random.normal(loc=5, scale=0.2, size=(n_geodesics,))
    a = np.random.normal(loc=2, scale=0.1, size=(n_geodesics,))
    b = np.random.normal(loc=10, scale=0.1, size=(n_geodesics,))

    geodesics = gs.zeros((n_geodesics, n_times, n_points, dim))
    times = gs.arange(0, 1, 1 / n_times)
    for i_geodesic in range(n_geodesics):
        start_circle = circle(n_points=n_points, radius=radii[i_geodesic])
        end_ellipse = ellipse(n_points=n_points, a=a[i_geodesic], b=b[i_geodesic])
        geodesic = METRIC.geodesic(initial_curve=start_circle, end_curve=end_ellipse)
        geodesics[i_geodesic] = geodesic(times)

    return geodesics

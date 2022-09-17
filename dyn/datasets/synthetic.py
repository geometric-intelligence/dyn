"""Utils to load synthetic datasets of cells."""

import geomstats.backend as gs
import numpy as np
import torch
from geomstats.geometry.discrete_curves import (  # , ClosedDiscreteCurves
    R2,
    DiscreteCurves,
    ElasticMetric,
)

# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures

CURVES_SPACE = DiscreteCurves(R2)
METRIC = CURVES_SPACE.srv_metric
# uncomment code below if you want to create geodesics with synthetic data
# using elastic metric
# CURVES_SPACE = DiscreteCurves(R2, a=1,b=10000)
# METRIC = CURVES_SPACE.elastic_metric


def rectangle(n_points_height, n_points_length, height, length, protusion=0):
    """Return a rectangle as an array of 2D points.

    This function returns a discrete closed curve representing a
    rectangle (height, length) as a array of its 2D points,
    in counter-clockwise order.
    """
    n_points = 2 * n_points_height + 2 * n_points_length - 4
    rectangle = gs.zeros((n_points, 2))

    height_pos_axis = height * gs.linspace(-1, 1, n_points_height)
    height_neg_axis = gs.flip(height_pos_axis, axis=0)

    length_pos_axis = length * gs.linspace(-1, 1, n_points_length)
    length_neg_axis = gs.flip(length_pos_axis, axis=0)

    heights = height * gs.ones(n_points_height)
    minus_heights = -heights
    lengths = length * gs.ones(n_points_length)
    minus_lengths = -lengths

    bottom = gs.vstack((length_pos_axis, minus_heights)).T[:-1]
    right = gs.vstack((lengths, height_pos_axis)).T[:-1]
    top = gs.vstack((length_neg_axis, heights)).T[:-1]
    left = gs.vstack((minus_lengths, height_neg_axis)).T[:-1]

    rectangle[: n_points_length - 1] = bottom
    rectangle[
        n_points_length - 1 : n_points_length + n_points_height - 2  # noqa: E203
    ] = right  # noqa: E203
    rectangle[
        n_points_length
        + n_points_height
        - 2 : 2 * n_points_length  # noqa: E203
        + n_points_height
        - 3
    ] = top
    rectangle[2 * n_points_length + n_points_height - 3 :] = left  # noqa: E203

    return rectangle


def rectangle_with_protusion(
    n_points_height, n_points_length, height, length, protusion_height
):
    """Return a rectangle with a protusion on its top side as an array of 2D points.

    The protusion simulates cells' protusions that appears during migration.

    This function returns a discrete closed curve representing a
    rectangle (height, length) as a array of its 2D points,
    in counter-clockwise order.
    """
    rect = rectangle(n_points_height, n_points_length, height, length)
    half_top_idx = n_points_length + n_points_height - 2 + n_points_length // 2
    protusion_half_length = n_points_length // 5
    protusion_start_idx = half_top_idx - protusion_half_length
    protusion_end_idx = half_top_idx + protusion_half_length
    rect[protusion_start_idx:protusion_end_idx] += protusion_height
    return rect


def ellipse(n_points, a, b):
    """Return an ellipse as an array of 2D points.

    This function returns a discrete closed curve representing the ellipse
    of equation: x**2/a**2 + y**2/b**2 =1.

    The discrete closed curve is represented by an array of 2D points
    in counter-clockwise order.

    The ellipse starts at the point (1, 0).

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


def ellipse_with_protusion(n_points, a, b, protusion_height):
    """Return an ellipse with a protusion on its top side as an array of 2D points.

    This function returns a discrete closed curve representing the ellipse
    of equation: x**2/a**2 + y**2/b**2 =1.
    The discrete closed curve is represented by an array of 2D points
    in counter-clockwise order.
    """
    ell = ellipse(n_points, a, b)
    half_top_idx = n_points // 4
    protusion_half_length = n_points // 16
    protusion_start_idx = half_top_idx - protusion_half_length
    protusion_end_idx = half_top_idx + protusion_half_length
    ell[protusion_start_idx:protusion_end_idx] += protusion_height
    return ell


def square(n_points_side, side=1):
    """Return a square as an array of 2D points.

    This function returns a discrete closed curve representing a
    unit square as a array of its 2D points, in counter-clockwise order.
    """
    return rectangle(n_points_side, n_points_side, side, side)


def square_with_protusion(n_points_side, side=1, protusion_height=1):
    """Return a square with a protusion on its top side as an array of 2D points.

    The protusion simulates cells' protusions that appears during migration.

    This function returns a discrete closed curve representing a
    rectangle (height, length) as a array of its 2D points,
    in counter-clockwise order.
    """
    return rectangle_with_protusion(
        n_points_side, n_points_side, side, side, protusion_height
    )


def circle(n_points, radius=1):
    """Return a circle as an array of 2D points.

    This function returns a discrete closed curve representing the
    unit circle, as an array of 2D points.
    """
    return ellipse(n_points, radius, radius)


def circle_with_protusion(n_points, radius=1, protusion_height=0):
    """Return a circle with a protusion on its top side as an array of 2D points.

    This function returns a discrete closed curve representing the
    unit circle, as an array of 2D points.
    """
    return ellipse_with_protusion(n_points, radius, radius, protusion_height)


def geodesics_square_to_rectangle(
    n_geodesics=10, n_times=20, n_points=40, protusion_height=0
):
    """Generate a dataset of geodesics that transform squares into rectangles."""
    dim = 2
    n_points_side = n_points_lengh = n_points_heigh = n_points // 4 + 1
    sides = np.random.normal(loc=1, scale=0.05, size=(n_geodesics,))
    heights = np.random.normal(loc=0.4, scale=0.05, size=(n_geodesics,))
    lengths = np.random.normal(loc=2, scale=0.05, size=(n_geodesics,))

    geodesics = gs.zeros((n_geodesics, n_times, n_points, dim))
    times = gs.arange(0, 1, 1 / n_times)
    for i_geodesic in range(n_geodesics):
        start_square = square_with_protusion(
            n_points_side=n_points_side,
            side=sides[i_geodesic],
            protusion_height=protusion_height,
        )
        end_rect = rectangle_with_protusion(
            n_points_height=n_points_heigh,
            n_points_length=n_points_lengh,
            height=heights[i_geodesic],
            length=lengths[i_geodesic],
            protusion_height=protusion_height,
        )
        geodesic = METRIC.geodesic(initial_curve=start_square, end_curve=end_rect)
        geodesics[i_geodesic] = geodesic(times)

    return geodesics


def geodesics_circle_to_ellipse(
    n_geodesics=10, n_times=20, n_points=40, protusion_height=0
):
    """Generate a dataset of geodesics that transform circles into ellipses."""
    dim = 2
    radii = np.random.normal(loc=1, scale=0.08, size=(n_geodesics,))
    a = np.random.normal(loc=2, scale=0.05, size=(n_geodesics,))
    b = np.random.normal(loc=0.5, scale=0.05, size=(n_geodesics,))

    geodesics = gs.zeros((n_geodesics, n_times, n_points, dim))
    times = gs.arange(0, 1, 1 / n_times)
    for i_geodesic in range(n_geodesics):
        start_circle = circle_with_protusion(
            n_points=n_points,
            radius=radii[i_geodesic],
            protusion_height=protusion_height,
        )
        end_ellipse = ellipse_with_protusion(
            n_points=n_points,
            a=a[i_geodesic],
            b=b[i_geodesic],
            protusion_height=protusion_height,
        )
        geodesic = METRIC.geodesic(start_circle, end_ellipse)
        geodesics[i_geodesic] = geodesic(times)

    return geodesics


def geodesic_between_curves(
    start_curve,
    end_curve,
    a,
    b,
    degree=1,
    n_times=20,
    n_sampling_points=40,
    noise_var=0,
):
    """Generate a trajectory between two real cell curves.

    Process used:
    - This notebook takes an input curve and and end curve
    - It then uses an f-transform to put these curves in a linear "q-space"
    - It draws a geodesic between these two curves in "q-space" (a line).
    - q = t * q1 + (1-t) * q2 is used to draw lines between points.
    - It samples from this geodesic in q-space.
    - Then, it transforms curves back into curve space using the inverese f
        transform.
    """
    #     CLOSED_CURVES_SPACE = ClosedDiscreteCurves(R2)

    elastic_metric = ElasticMetric(a, b, ambient_manifold=R2)

    q_start = elastic_metric.f_transform(start_curve)
    q_end = elastic_metric.f_transform(end_curve)

    times = gs.arange(0, n_times, 1)

    q_trajectory = []
    for time in times:
        q_trajectory.append(time * q_start + (1 - time) * q_end)

    q_trajectory = np.array(q_trajectory)

    q_trajectory = torch.from_numpy(q_trajectory)

    starting_point_array = gs.zeros((n_times, 2))

    curves = elastic_metric.f_transform_inverse(q_trajectory, starting_point_array)

    #     closed_curves = CLOSED_CURVES_SPACE.projection(curves)

    if noise_var == 0.0:
        return curves
    else:
        print("Error: Different noise levels not implemented")


# def trajectory_between_curves_regression(
#     start_curve, end_curve, a, b, degree=1, n_times=20, n_points=40
# ):
#     """Generate a synthetic trajectory between cells using regression.

#     Process used:
#     - This notebook takes an input curve and and end curve
#     - It then uses an f-transform to put these curves in a linear "q-space"
#     - It draws a geodesic between these two curves in "q-space" (a line).
#     - q = t * q1 + (1-t) * q2 is used to draw lines between points.
#     - It samples from this geodesic in q-space.
#     - Then, it transforms curves back into curve space using the inverese f
#         transform.
#     """
#     elastic_metric = ElasticMetric(a, b, ambient_manifold=R2)

#     q = elastic_metric.f_transform(gs.stack([start_curve,end_curve], axis = -1))

#     q_vector = q.reshape((2, -1))

#     times = gs.arange(0, n_times, 1)
#     times = np.reshape(times, (n_times, 1))

#     x = np.reshape(np.array([0, n_times-1]), (2, 1))
#     y = q_vector

#     polynomial_features = PolynomialFeatures(degree=degree)
#     x_poly = polynomial_features.fit_transform(x)

#     model = LinearRegression()
#     model.fit(x_poly, y)
#     y_poly_pred = model.predict(x_poly)

#     q_trajectory = torch.from_numpy(y_poly_pred)

#     starting_point_array = gs.zeros((len(q_trajectory), 2))

#     curves = elastic_metric.f_transform_inverse(q_trajectory, starting_point_array)

#     return curves

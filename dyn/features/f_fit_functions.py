"""Functions for geodesic and non-geodesic regression and metric learning."""

import os

import geomstats.backend as gs
import matplotlib.pyplot as plt
import numpy as np
import torch

# load discrete curves and R2 manifolds
from geomstats.geometry.discrete_curves import (
    R2,
    ClosedDiscreteCurves,
    DiscreteCurves,
    ElasticMetric,
)

# from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.pre_shape import PreShapeSpace
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# import dyn.dyn.features.basic as basic
os.environ["GEOMSTATS_BACKEND"] = "pytorch"


def ftrans_plot_predictions(geodesic, a=1, b=1, split=True):
    """Compute the best fit geodesic by transforming data into q space.

    Then plots the geodesic prediction and the original data.

    Intended for geodesics

    If split is false:
    Uses the f transform described in http://arxiv.org/abs/1803.10894 to:
    -transform data into "q-space" which is flat.
    -perform linear regression in this "q-space".
    -calculate the "predicted data" that the line in "q-space" predicts
    -here, the "predicted data" is "where the regression line would predict the
        original input data to be"
    -transform predicted data back to curve space
    -plot 1) the original data 2) the predicted data 3) the predicted data and
        the original data, overlaid.

    If split is true:
    Does the same thing except:
    -it uses the first set of the dataset to generate regression prediction
    -then, it uses the regression line to "predict" the second set of the time series
    -then, plots: 1) the predicted second half of the time series 2) the actual second
        set of the time series
    3) the predicted and actual second set of the time series, overlaid.


    Parameters:
    ----------
    -geodesic: a single time series (the name "geodesic" is used because we are hoping
    that the curves in the time series will follow geodesics.
    -a and b are parameters for the elastic metric.
    -split describes whether the dataset (time series) will be split up into a
    "training" dataset (to train the regression line) and a "test" dataset OR whether
    the entire dataset will be used to train the regression line.
    """
    # NEW, FOR CLOSED CURVES:
    # R1 = Euclidean(dim=1)
    CLOSED_CURVES_SPACE = ClosedDiscreteCurves(R2)
    # CURVES_SPACE = DiscreteCurves(R2)

    # n_geodesics = 1
    n_times = len(geodesic)
    n_points = len(geodesic[0])

    # instate an object elastic_metric of the class ElasticMetric
    elastic_metric = ElasticMetric(a, b, ambient_manifold=R2)

    # q_tensor = elastic_metric.f_transform(geods_circle_ell[0])
    q_tensor = elastic_metric.f_transform(geodesic)

    q = np.array(q_tensor)

    # reshape q into a compressed vector.
    q_vector = q.reshape((n_times, -1))

    # create regression object. this should probably be in the regression function.
    # regr = linear_model.LinearRegression()

    # Now, i need to create an array that only has the times
    q_times_1d = gs.arange(0, n_times, 1)

    if split:
        q_tensor_predict, starting_point_array = half_set_linear_regression(
            q_vector, n_times, q_times_1d, n_points
        )

    else:
        q_tensor_predict, starting_point_array = full_set_linear_regression(
            q_vector, n_times, q_times_1d, n_points
        )

    predicted_curves = elastic_metric.f_transform_inverse(
        q_tensor_predict, starting_point_array
    )

    # first, i will create a new array, where one of the geodesics is the original
    # geodesic and the other geodesic
    # is the predicted geodesic. The third figure will show the two curves overlaid
    # on each other.

    recentered_curves = centered_predictions(predicted_curves, n_points, n_times)

    # NEW
    closed_recentered_curves = CLOSED_CURVES_SPACE.projection(recentered_curves)

    if split:
        half_set_plot_comparison(closed_recentered_curves, geodesic, n_times, a, b)
    else:
        full_set_plot_comparison(closed_recentered_curves, geodesic, n_times, a, b)


def full_set_linear_regression(q_vector, n_times, q_times_1d, n_points):
    """Perform a linear regression in "q-space".

    Training dataset is the whole time series.

    Parameters:
    ----------
    -q_vector: this is the set of curves that have been transformed into "q's" so
        that we can do operations in q space.
    -n_times: the number of curves (times) in the time series
    -n_points: the number of points in a curve
    -q_times_1d: an array that holds a set of time points (the set of time points
        where each curve was recorded). this isa 1d array but will need to be reshaped
        so that we can use it for regression.
    """
    # create regression object
    regr = linear_model.LinearRegression()

    q_times = np.reshape(q_times_1d, (n_times, 1))

    regr.fit(q_times, q_vector)

    # compute estimated q predictions
    q_vector_predict = regr.predict(q_times)

    # first, turn the vector back into its original shape)
    q_array_predict = np.reshape(q_vector_predict, (n_times, n_points - 1, 2))

    # transform the array back into a tensor s.t, f_transoform_inverse will accept it
    q_tensor_predict = torch.from_numpy(q_array_predict)

    starting_point_array = gs.zeros((n_times, 2))

    return q_tensor_predict, starting_point_array


def half_set_linear_regression(q_vector, n_times, q_times_1d, n_points):
    """Perform a linear regression in "q-space".

    First half of the dataset isused to "train" the regression line and the second
        half of the dataset is used to "test" the regression line.

    Parameters:
    ----------
    -q_vector: this is the set of curves that have been transformed into "q's" so that
        we can do operations in q space.
    -n_times: the number of curves (times) in the time series
    -n_points: the number of points in a curve
    -q_times_1d: an array that holds a set of time points (the set of time points where
        each curve was recorded). this isa 1d array but will need to be reshaped so that
         we can use it for regression.
    """
    # create regression object. this should probably be in the regression function.
    regr = linear_model.LinearRegression()

    half_n_times = int(n_times / 2)

    # NEW HERE: splitting the dataset
    q_times_1d_train = q_times_1d[:half_n_times]
    q_times_1d_test = q_times_1d[half_n_times:]

    # splitting the vector dataset
    q_vector_train = q_vector[:half_n_times]
    # q_vector_test = q_vector[half_n_times:]

    q_times_train = np.reshape(q_times_1d_train, (half_n_times, 1))
    q_times_test = np.reshape(q_times_1d_test, (n_times - half_n_times, 1))

    regr.fit(q_times_train, q_vector_train)

    # compute estimated q predictions
    q_vector_predict = regr.predict(q_times_test)

    q_array_predict = np.reshape(
        q_vector_predict, (n_times - half_n_times, n_points - 1, 2)
    )

    # now,transform the array back into a tensor s.t f_transoform_inverse will accept it
    q_tensor_predict = torch.from_numpy(q_array_predict)

    # do the transform
    starting_point_array = gs.zeros((n_times - half_n_times, 2))

    return q_tensor_predict, starting_point_array


def half_set_plot_comparison(recentered_curves, geodesic, n_times, a, b):
    """Plot the data curves, the predicted curves, and the two, overlaid.

    Considers the case where the first half of the dataset is used as a "training"
        dataset and the second half of the dataset is used as a "testing" dataset.

    Plots three figures:
    1) the second half of the original geodesic
    2) the second half of the original geodesic, AS PREDICTED BY THE REGRESSION LINE
    3) the first two plots, overlaid.

    Parameters:
    - recentered curves: the curves predicted by the regression line, centered at
        their mean point.
    - geodesic: the original geodesic that we are doing analysis on
    - n_times: the number of time points in the geodesic (number of curves in the
        time series)
    - a and b: elastic metric parameters

    """
    half_n_times = int(n_times / 2)

    geodesic_array = np.array([geodesic[half_n_times:], recentered_curves])

    n_geodesics_plot = 2
    fig, axes = plt.subplots(
        n_geodesics_plot + 1,
        n_times - half_n_times,
        figsize=(20, 10),  # sharex=True, sharey=True
    )
    fig.suptitle(
        "Elastic Metric: a= "
        + str(a)
        + ", b= "
        + str(b)
        + ": Comparison between synthetic and 'q-predicted' geodesics",
        fontsize=20,
    )

    for i_geodesic in range(n_geodesics_plot):
        curve = geodesic_array[i_geodesic]
        for i_time in range(n_times - half_n_times):
            axes[i_geodesic, i_time].plot(
                curve[i_time][:, 0], curve[i_time][:, 1], marker="o", c=f"C{i_geodesic}"
            )
            axes[i_geodesic, i_time].set_aspect("equal")

    # now, creating the third set of plots, where they are overlaid
    for i_geodesic in range(n_geodesics_plot):
        curve = geodesic_array[i_geodesic]
        for i_time in range(n_times - half_n_times):
            axes[2, i_time].plot(
                curve[i_time][:, 0], curve[i_time][:, 1], marker="o", c=f"C{i_geodesic}"
            )
            axes[i_geodesic, i_time].set_aspect("equal")

    plt.tight_layout()


def full_set_plot_comparison(recentered_curves, geodesic, n_times, a, b):
    """Plot the data curves, the predicted curves, and the two, overlaid.

    Considers the case where the whole dataset is used to train the regression line,
    and the regression line, and the original geodesic is compared against the points
    that lie on the regression line. The regression line is a geodesic in q space,
    so if the regression line predictions perfectly match the original geodesic, that
    confirms that the original geodesic is actually a geodesic.

    Plots three figures:
    1) the original geodesic
    2) the original geodesic, AS PREDICTED BY THE REGRESSION LINE
    3) the first two plots, overlaid.

    Parameters:
    - recentered curves: the curves predicted by the regression line, centered at their
        mean point.
    - geodesic: the original geodesic that we are doing analysis on
    - n_times: the number of time points in the geodesic (number of curves in the time
        series)
    - a and b: elastic metric parameters

    """
    geodesic_array = np.array([geodesic, recentered_curves])

    n_geodesics_plot = 2
    fig, axes = plt.subplots(
        n_geodesics_plot + 1, n_times, figsize=(20, 10), sharex=True, sharey=True
    )
    fig.suptitle(
        "Elastic Metric: a= "
        + str(a)
        + ", b= "
        + str(b)
        + ": Comparison between synthetic and 'q-predicted' geodesics",
        fontsize=20,
    )

    for i_geodesic in range(n_geodesics_plot):
        curve = geodesic_array[i_geodesic]
        for i_time in range(n_times):
            axes[i_geodesic, i_time].plot(
                curve[i_time][:, 0], curve[i_time][:, 1], marker="o", c=f"C{i_geodesic}"
            )
            axes[i_geodesic, i_time].set_aspect("equal")

    # now, creating the third set of plots, where they are overlaid
    for i_geodesic in range(n_geodesics_plot):
        curve = geodesic_array[i_geodesic]
        for i_time in range(n_times):
            axes[2, i_time].plot(
                curve[i_time][:, 0], curve[i_time][:, 1], marker="o", c=f"C{i_geodesic}"
            )
            axes[i_geodesic, i_time].set_aspect("equal")

    plt.tight_layout()


def centered_predictions(predicted_curves, n_points, n_times):
    """Center the shapes on their centers of mass."""
    n_sampling_points = n_points
    cell_centers = gs.zeros((n_times, 2))
    cell_shapes = gs.zeros((n_times, n_sampling_points, 2))

    for i_contour, contour in enumerate(predicted_curves):
        interpolated = _interpolate(contour, n_sampling_points)
        cleaned = _remove_consecutive_duplicates(interpolated)
        center = gs.mean(cleaned, axis=-2)
        centered = cleaned - center[..., None, :]
        cell_centers[i_contour] = center
        cell_shapes[i_contour] = centered

    # for i_cell, cell in enumerate(cell_shapes):
    #    cell_shapes[i_cell] = cell / basic.perimeter(cell_shapes[i_cell])

    # for i_cell, cell_shape in enumerate(cell_shapes):
    #    cell_shapes[i_cell] = _exhaustive_align(cell_shape, cell_shapes[0])

    return cell_shapes


def _interpolate(curve, n_sampling_points):
    """Interpolate a discrete curve with nb_points from a discrete curve.

    Parameters
    ----------
    curve : array-like, shape=[n_points, 2]
    n_sampling_points : int

    Returns
    -------
    interpolation : array-like, shape=[n_sampling_points, 2]
       Discrete curve with n_sampling_points
    """
    old_length = curve.shape[0]
    interpolation = np.zeros((n_sampling_points, 2))
    incr = old_length / n_sampling_points
    pos = np.array(0.0, dtype=np.float32)
    for i in range(n_sampling_points):
        index = int(np.floor(pos))
        interpolation[i] = curve[index] + (pos - index) * (
            curve[(index + 1) % old_length] - curve[index]
        )
        pos += incr
    return gs.array(interpolation, dtype=gs.float32)


def _remove_consecutive_duplicates(curve, tol=1e-2):
    """Preprocess curve to ensure that there are no consecutive duplicate points.

    Returns
    -------
    curve : discrete curve
    """
    dist = curve[1:] - curve[:-1]
    dist_norm = gs.sqrt(gs.sum(dist**2, axis=1))

    if gs.any(dist_norm < tol):
        for i in range(len(curve) - 1):
            if gs.sqrt(gs.sum((curve[i + 1] - curve[i]) ** 2, axis=0)) < tol:
                curve[i + 1] = (curve[i] + curve[i + 2]) / 2

    return curve


def _exhaustive_align(curve, base_curve):
    """Project a curve in shape space.

    This happens in 2 steps:
    - remove translation (and scaling?) by projecting in pre-shape space.
    - remove rotation by exhaustive alignment minimizing the L² distance.

    Returns
    -------
    aligned_curve : discrete curve
    """
    M_AMBIENT = 2

    n_sampling_points = curve.shape[-2]
    preshape = PreShapeSpace(m_ambient=M_AMBIENT, k_landmarks=n_sampling_points)

    nb_sampling = len(curve)
    distances = gs.zeros(nb_sampling)
    for shift in range(nb_sampling):
        reparametrized = gs.array(
            [curve[(i + shift) % nb_sampling] for i in range(nb_sampling)]
        )
        aligned = preshape.align(point=reparametrized, base_point=base_curve)
        distances[shift] = preshape.embedding_metric.norm(
            gs.array(aligned) - gs.array(base_curve)
        )
    shift_min = gs.argmin(distances)
    reparametrized_min = gs.array(
        [curve[(i + shift_min) % nb_sampling] for i in range(nb_sampling)]
    )
    aligned_curve = preshape.align(point=reparametrized_min, base_point=base_curve)
    return aligned_curve


def optimize_ab_linear(geodesic):
    """Calculate optimal a, b params using linear regression and R2 score as a test.

    could probably make this function more "sophisticated" by implementing something
    that would
    1) see which values give the best result
    2) implement some algorithm to narrow down a more precise "best value"

    Note: this uses the whole dataset to do the optimization and does not split the
    dataset
    """
    # n_geodesics = 1
    n_times = len(geodesic)
    # n_points = len(geodesic[0])

    regr = linear_model.LinearRegression()

    ELASTIC_METRIC = {}
    AS = [1, 2, 0.75, 0.5, 0.25, 0.01, 1.6, 1.4, 1.2, 1, 0.5, 0.2, 0.1]
    BS = [0.5, 1, 0.5, 0.5, 0.5, 0.5, 2, 2, 2, 2, 2, 2, 2]

    max_r2 = 0
    best_a = 0
    best_b = 0

    for a, b in zip(AS, BS):
        ELASTIC_METRIC[a, b] = DiscreteCurves(R2, a=a, b=b).elastic_metric
        q_tensor = ELASTIC_METRIC[a, b].f_transform(geodesic)
        q = np.array(q_tensor)
        q_vector = q.reshape((n_times, -1))

        q_times_1d = gs.arange(0, n_times, 1)
        q_times = np.reshape(q_times_1d, (n_times, 1))

        regr.fit(q_times, q_vector)
        r2 = regr.score(q_times, q_vector)

        if r2 > max_r2:
            max_r2 = r2
            best_a = a
            best_b = b

    return best_a, best_b, max_r2


def rmse(geodesic, a, b, degree=2, split=True):
    """Return root mean squared error for a given geodesic, a, b, degree."""
    # CLOSED_CURVES_SPACE = ClosedDiscreteCurves(R2)

    n_times = len(geodesic)
    n_points = len(geodesic[0])

    print(geodesic.shape)

    # instates an object elastic_metric of the class ElasticMetric
    elastic_metric = ElasticMetric(a, b, ambient_manifold=R2)

    # q_tensor = elastic_metric.f_transform(geods_circle_ell[0])
    q_tensor = elastic_metric.f_transform(geodesic)

    q = np.array(q_tensor)
    print(q.shape)

    # reshape q into a compressed vector.
    q_vector = q.reshape((n_times, -1))

    # Now, i need to create an array that only has the times
    q_times_1d = gs.arange(0, n_times, 1)

    if split:
        # ToDo: don't pass n_times. instead, compute n_times from len(q_times_1d).
        # n_points is second dimension of q_vector divided by 2 plus 1.
        q_tensor_predict, starting_point_array, rmse = polynomial_regression(
            q_vector, n_times, q_times_1d, n_points, degree
        )

    else:
        q_tensor_predict, starting_point_array, rmse = polynomial_regression(
            q_vector, n_times, q_times_1d, n_points, degree
        )

    return rmse


def ftrans_plot_predictions_nongeodesic(geodesic, a, b, degree=2, split=True):
    """Perform non-geodesic regression and plot results against data.

    This will look a lot like ftrans_plot_predictions except it will do polynomial
    regression instead of linear regression.
    I think we should also include a function here that decides which polynomial
    function to do (by optimizing r^2). Then, we can use this, (in conjunction with
    a future optimize_ab_polynomial) to fully fit functions In the future, we can
    also combine all of these functions so that we can have one big function that
    has a bunch of options like type = "linear", "polynomial" etc.
    """
    # NEW, FOR CLOSED CURVES:
    # R1 = Euclidean(dim=1)
    CLOSED_CURVES_SPACE = ClosedDiscreteCurves(R2)
    # CURVES_SPACE = DiscreteCurves(R2)

    # n_geodesics = 1
    n_times = len(geodesic)
    n_points = len(geodesic[0])

    print(geodesic.shape)

    # instates an object elastic_metric of the class ElasticMetric
    elastic_metric = ElasticMetric(a, b, ambient_manifold=R2)

    # q_tensor = elastic_metric.f_transform(geods_circle_ell[0])
    q_tensor = elastic_metric.f_transform(geodesic)

    q = np.array(q_tensor)
    print(q.shape)

    # reshape q into a compressed vector.
    q_vector = q.reshape((n_times, -1))

    # Now, i need to create an array that only has the times
    q_times_1d = gs.arange(0, n_times, 1)

    if split:
        # ToDo: don't pass n_times. instead, compute n_times from len(q_times_1d).
        # n_points is second dimension of q_vector divided by 2 plus 1.
        q_tensor_predict, starting_point_array, rmse = polynomial_regression(
            q_vector, n_times, q_times_1d, n_points, degree
        )

    else:
        q_tensor_predict, starting_point_array, rmse = polynomial_regression(
            q_vector, n_times, q_times_1d, n_points, degree
        )

    predicted_curves = elastic_metric.f_transform_inverse(
        q_tensor_predict, starting_point_array
    )

    # first, i will create a new array, where one of the geodesics is the original
    # geodesic and the other geodesic
    # is the predicted geodesic. The third figure will show the two curves overlaid
    # on each other.

    recentered_curves = centered_predictions(predicted_curves, n_points, n_times)

    closed_recentered_curves = CLOSED_CURVES_SPACE.projection(recentered_curves)

    if split:
        half_set_plot_comparison(closed_recentered_curves, geodesic, n_times, a, b)
    else:
        full_set_plot_comparison(closed_recentered_curves, geodesic, n_times, a, b)


def polynomial_regression(q_vector, n_times, q_times_1d, n_points, deg):
    """Perform polynomial regression in q space."""
    q_times = np.reshape(q_times_1d, (n_times, 1))

    x = q_times
    y = q_vector

    polynomial_features = PolynomialFeatures(degree=deg)
    x_poly = polynomial_features.fit_transform(x)

    model = LinearRegression()
    model.fit(x_poly, y)
    y_poly_pred = model.predict(x_poly)

    rmse = np.sqrt(mean_squared_error(y, y_poly_pred))
    r2 = r2_score(y, y_poly_pred)
    print(rmse)
    print(r2)

    # de-compress the vector (turn it back into its original shape)
    y_poly_pred_array = np.reshape(y_poly_pred, (n_times, n_points - 1, 2))

    # transform the array back into a tensor s.t. f_transoform_inverse will accept it
    y_poly_pred_tensor = torch.from_numpy(y_poly_pred_array)

    starting_point_array = gs.zeros((n_times, 2))

    return y_poly_pred_tensor, starting_point_array, rmse


def half_set_polynomial_regression(q_vector, n_times, q_times_1d, n_points, deg):
    """Perform polynomial regression in q space, using half of the dataset to train.

    Parameters:
    ----------
    -q_vector: this is the set of curves that have been transformed into "q's" so that
        we can do operations in q space.
    -n_times: the number of curves (times) in the time series
    -n_points: the number of points in a curve
    -q_times_1d: an array that holds a set of time points (the set of time points where
        each curve was recorded). this is a 1d array but will need to be reshaped so
        that we can use it for regression.
    """
    # create regression object. this should probably be in the regression function.
    # regr = linear_model.LinearRegression()

    half_n_times = int(n_times / 2)

    # NEW HERE: splitting the dataset
    q_times_1d_train = q_times_1d[:half_n_times]
    q_times_1d_test = q_times_1d[half_n_times:]

    # splitting the vector dataset
    q_vector_train = q_vector[:half_n_times]
    # q_vector_test = q_vector[half_n_times:]

    q_times_train = np.reshape(q_times_1d_train, (half_n_times, 1))
    q_times_test = np.reshape(q_times_1d_test, (n_times - half_n_times, 1))

    x = q_times_train
    y = q_vector_train

    polynomial_features = PolynomialFeatures(degree=deg)
    x_poly = polynomial_features.fit_transform(x)

    model = LinearRegression()
    model.fit(x_poly, y)
    y_poly_pred = model.predict(q_times_test)

    rmse = np.sqrt(mean_squared_error(y, y_poly_pred))
    r2 = r2_score(y, y_poly_pred)
    print(rmse)
    print(r2)

    # de-compress the vector (turn it back into its original shape)
    y_poly_pred_array = np.reshape(
        y_poly_pred, (n_times - half_n_times, n_points - 1, 2)
    )

    # transform the array back into a tensor s.t. f_transoform_inverse will accept it
    y_poly_pred_tensor = torch.from_numpy(y_poly_pred_array)

    # regr.fit(q_times_train,q_vector_train)

    # compute estimated q predictions
    # q_vector_predict=regr.predict(q_times_test)

    # q_array_predict = np.reshape(q_vector_predict,(n_times-half_n_times,n_points-1,2))

    # transform the array back into a tensor s.t. f_transoform_inverse will accept it

    # q_tensor_predict= torch.from_numpy(q_array_predict)

    # do the transform
    starting_point_array = gs.zeros((n_times - half_n_times, 2))

    return y_poly_pred_tensor, starting_point_array, rmse

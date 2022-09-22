"""Unit tests for the optimization on a and m."""

import geomstats.backend as gs
from geomstats.geometry.discrete_curves import R2, ElasticMetric

import dyn.dyn.datasets.synthetic as synthetic
import dyn.dyn.features.optimize_am as optimize_am


def test_tau_matrix():
    """Test the computation of the tau_jl."""
    m = 1
    times_train = [1, 3, 4]
    X = gs.array([[1, 1], [1, 3], [1, 4]])
    assert X.shape == (3, m + 1)
    expected_tau_mat = gs.linalg.inv(X.T @ X) @ X.T

    assert expected_tau_mat.shape == (m + 1, 3), expected_tau_mat.shape

    result_tau_mat = optimize_am.tau_matrix(times_train, m)

    print("expected", expected_tau_mat)
    print("result", result_tau_mat)
    assert gs.allclose(result_tau_mat, expected_tau_mat)


def test_r_squared():
    """Test the computation of the R^2."""
    curve_trajectories = synthetic.geodesics_circle_to_ellipse(
        n_geodesics=2, n_times=25, n_points=50
    )
    a_true = 1  # geodesics_circle_to_ellipse uses SRV metric, thus a = 1
    m_true = 1  # geodesics_circle_to_ellipse creates geodesics
    times_train = gs.arange(0, 10, 1)
    times_val = gs.arange(10, 25, 1)
    print(times_train)
    print(times_val)

    one_trajectory = curve_trajectories[0]
    result = optimize_am.r_squared(
        one_trajectory, times_train, times_val, degree=m_true, a=a_true
    )

    assert gs.allclose(result, 1.0), result


def test_mse():
    """Test the computation of the MSE."""
    curve_trajectories = synthetic.geodesics_circle_to_ellipse(
        n_geodesics=2, n_times=25, n_points=50
    )
    a_true = 1  # geodesics_circle_to_ellipse uses SRV metric, thus a = 1
    m_true = 1  # geodesics_circle_to_ellipse creates geodesics

    b = 0.5
    elastic_metric = ElasticMetric(a_true, b, ambient_manifold=R2)

    times_train = gs.arange(0, 10, 1)
    times_val = gs.arange(10, 25, 1)
    print(times_train)
    print(times_val)

    one_trajectory = curve_trajectories[0]
    elastic_metric = elastic_metric
    result = optimize_am.mse(
        trajectory=one_trajectory,
        elastic_metric=elastic_metric,
        times_train=times_train,
        times_val=times_val,
        degree=m_true,
        a=a_true,
    )

    assert gs.allclose(result, 0.0), result

"""Unit tests for the optimization on a and m."""

import geomstats.backend as gs
from geomstats.geometry.discrete_curves import R2, ElasticMetric

import dyn.dyn.datasets.synthetic as synthetic
import dyn.dyn.features.optimize_am as optimize_am


def test_r_squared():
    curve_trajectories = synthetic.geodesics_circle_to_ellipse(
            n_geodesics=2, n_times=25, n_points=50)
    a_true = 1  # geodesics_circle_to_ellipse uses SRV metric, thus a = 1
    m_true = 1  # geodesics_circle_to_ellipse creates geodesics
    times_train = gs.arange(0, 10, 1)
    times_val = gs.arange(10, 25, 1)
    print(times_train)
    print(times_val)
    
    one_trajectory = curve_trajectories[0]
    result = optimize_am.r_squared(
        one_trajectory, times_train, times_val, degree=m_true, a=a_true)

    assert gs.allclose(result, 1.0), result


def test_mse():
    curve_trajectories = synthetic.geodesics_circle_to_ellipse(
            n_geodesics=2, n_times=25, n_points=50)
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
        curve_trajectory=one_trajectory, elastic_metric=elastic_metric,
        times_train=times_train, times_val=times_val, degree=m_true, a=a_true)

    assert gs.allclose(result, 0.0), result

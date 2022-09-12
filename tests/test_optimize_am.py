"""Unit tests for the optimization on a and m."""

import geomstats.backend as gs

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

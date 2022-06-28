"""Test functions in the synthetic module."""

import dyn.datasets.synthetic as synthetic


def test_rectangle():
    """Test that function rectangle runs and outputs correct shape."""
    rect = synthetic.rectangle(10, 10, 2, 3)
    assert rect.shape == (36, 2), rect.shape


def test_rectangle_with_protusion():
    """Test that function rectangle_with_protusion runs and outputs correct shape."""
    rect = synthetic.rectangle_with_protusion(10, 10, 2, 3, 3)
    assert rect.shape == (36, 2), rect.shape


def test_square():
    """Test that function square runs and outputs correct shape."""
    square = synthetic.square(10)
    assert square.shape == (36, 2), square.shape


def test_square_with_protusion():
    """Test that function square_with_protusion runs and outputs correct shape."""
    square = synthetic.square_with_protusion(10, protusion_height=2)
    assert square.shape == (36, 2), square.shape


def test_ellipse():
    """Test that function ellipse runs and outputs correct shape."""
    ell = synthetic.ellipse(20, 2, 3)
    assert ell.shape == (20, 2), ell.shape


def test_ellipse_with_protusion():
    """Test that function ellipse_with_protusion runs and outputs correct shape."""
    ell = synthetic.ellipse_with_protusion(20, 2, 3, protusion_height=2)
    assert ell.shape == (20, 2), ell.shape


def test_circle():
    """Test that function circle runs and outputs correct shape."""
    circle = synthetic.circle(20)
    assert circle.shape == (20, 2), circle.shape


def test_circle_with_protusion():
    """Test that function circle_with_protusion runs and outputs correct shape."""
    circle = synthetic.circle_with_protusion(20, protusion_height=2)
    assert circle.shape == (20, 2), circle.shape


def test_geodesics_square_to_rectangle():
    """Test that function geodesics_square_to_rectangle runs.

    And outputs correct shape.
    """
    geodesics = synthetic.geodesics_square_to_rectangle(
        n_geodesics=10, n_times=20, n_points=40
    )
    assert geodesics.shape == (10, 20, 40, 2), geodesics.shape


def test_geodesics_circle_to_ellipse():
    """Test that function geodesics_circle_to_ellipse runs.

    And outputs correct shape.
    """
    geodesics = synthetic.geodesics_circle_to_ellipse(
        n_geodesics=10, n_times=20, n_points=40
    )
    assert geodesics.shape == (10, 20, 40, 2), geodesics.shape

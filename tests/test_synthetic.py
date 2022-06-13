"""Test functions in the synthetic module."""

import cells.datasets.synthetic as synthetic


def test_rectangle():
    """Test that function rectangle runs and outputs correct shape."""
    rect = synthetic.rectangle(10, 10, 2, 3)
    assert rect.shape == (36, 2), rect.shape


def test_square():
    """Test that function unit_square runs and outputs correct shape."""
    square = synthetic.unit_square(10)
    assert square.shape == (36, 2), square.shape


def test_ellipse():
    """Test that function ellipse runs and outputs correct shape."""
    ell = synthetic.ellipse(20, 2, 3)
    assert ell.shape == (20, 2), ell.shape


def test_circle():
    """Test that function unit_circle runs and outputs correct shape."""
    circle = synthetic.unit_circle(20)
    assert circle.shape == (20, 2), circle.shape

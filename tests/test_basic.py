"""Unit tests for the basic shape features."""

import geomstats.backend as gs

import dyn.features.basic as basic


def test_perimeter():
    """Test that the perimeter of a rectangle gives expected result."""
    rectangle = gs.array([[2, 1], [-2, 1], [-2, -1], [2, -1]])
    result = basic.perimeter(rectangle)
    expected = 12
    assert result == expected


def test_area():
    """Test that the area of a rectangle gives expected result."""
    rectangle = gs.array([[2, 1], [-2, 1], [-2, -1], [2, -1]])
    result = basic.area(rectangle)
    expected = 8
    assert result == expected

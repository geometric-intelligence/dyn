"""Test functions in the experimental module."""

import cells.datasets.experimental as experimental


def test_load_treated_osteosarcoma_cells():
    """Test that function load_treated_osteosarcoma_cells runs.

    And outputs correct shape.
    """
    n_sampling_points = 10

    cells, lines, treatments = experimental.load_treated_osteosarcoma_cells(
        n_sampling_points=n_sampling_points
    )
    assert cells.shape == (650, n_sampling_points, 2), cells.shape
    assert len(lines) == 650, len(lines)
    assert len(treatments) == 650, len(treatments)


def test_load_mutated_retinal_cells():
    """Test that function load_mutated_retinal_cells runs.

    And outputs correct shape.
    """
    n_sampling_points = 10
    cells, surfaces, mutations = experimental.load_mutated_retinal_cells(
        n_sampling_points=n_sampling_points
    )
    assert cells.shape == (3871, n_sampling_points, 2), cells.shape
    assert len(surfaces) == 3871, len(surfaces)
    assert len(mutations) == 3871, len(mutations)

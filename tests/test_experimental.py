"""Test functions in the experimental module."""

import cells.datasets.experimental as experimental


def test_load_treated_osteosarcoma_cells():
    """Test that function load_treated_osteosarcoma_cells runs.

    And outputs correct shape.
    """
    n_sampling_points = 10
    n_cells = 5

    cells, lines, treatments = experimental.load_treated_osteosarcoma_cells(
        n_cells=n_cells, n_sampling_points=n_sampling_points
    )
    assert cells.shape == (n_cells, n_sampling_points, 2), cells.shape
    assert len(lines) == n_cells, len(lines)
    assert len(treatments) == n_cells, len(treatments)


def test_load_mutated_retinal_cells():
    """Test that function load_mutated_retinal_cells runs.

    And outputs correct shape.
    """
    n_sampling_points = 10
    n_cells = 5

    cells, surfaces, mutations = experimental.load_mutated_retinal_cells(
        n_cells=n_cells, n_sampling_points=n_sampling_points
    )
    assert cells.shape == (n_cells, n_sampling_points, 2), cells.shape
    assert len(surfaces) == n_cells, len(surfaces)
    assert len(mutations) == n_cells, len(mutations)

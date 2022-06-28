"""Test functions in the experimental module."""

import numpy as np

import dyn.datasets.experimental as experimental
import dyn.features.basic as basic


def test_load_treated_osteosarcoma_cells():
    """Test that function load_treated_osteosarcoma_cells runs.

    And outputs correct shape.
    """
    n_sampling_points = 10
    n_cells = 5

    (
        cells,
        cell_shapes,
        lines,
        treatments,
    ) = experimental.load_treated_osteosarcoma_cells(
        n_cells=n_cells, n_sampling_points=n_sampling_points
    )
    assert cells.shape == (n_cells, n_sampling_points, 2), cells.shape
    assert len(lines) == n_cells, len(lines)
    assert len(treatments) == n_cells, len(treatments)
    for cell in cell_shapes:
        assert np.allclose(basic.perimeter(cell), 1.0)


def test_load_mutated_retinal_cells():
    """Test that function load_mutated_retinal_cells runs.

    And outputs correct shape.
    """
    n_sampling_points = 10
    n_cells = 5

    cells, cell_shapes, surfaces, mutations = experimental.load_mutated_retinal_cells(
        n_cells=n_cells, n_sampling_points=n_sampling_points
    )
    assert cells.shape == (n_cells, n_sampling_points, 2), cells.shape
    assert len(surfaces) == n_cells, len(surfaces)
    assert len(mutations) == n_cells, len(mutations)
    for cell in cell_shapes:
        assert np.allclose(basic.perimeter(cell), 1.0)

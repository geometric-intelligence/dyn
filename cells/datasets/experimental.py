"""Utils to load experimental datasets of cells."""


import geomstats.backend as gs
import geomstats.datasets.utils as data_utils


def _interpolate(curve, n_sampling_points):
    """Interpolate a discrete curve with nb_points from a discrete curve.

    Parameters
    ----------
    curve :
    n_sampling_points : int

    Returns
    -------
    interpolation : discrete curve with nb_points points
    """
    old_length = curve.shape[0]
    interpolation = gs.zeros((n_sampling_points, 2))
    incr = old_length / n_sampling_points
    pos = 0
    for i in range(n_sampling_points):
        index = int(gs.floor(pos))
        interpolation[i] = curve[index] + (pos - index) * (
            curve[(index + 1) % old_length] - curve[index]
        )
        pos += incr
    return interpolation


def _remove_consecutive_duplicates(curve, tol=1e-10):
    """Preprocess curve to ensure that there are no consecutive duplicate points.

    Returns
    -------
    curve : discrete curve
    """
    dist = curve[1:] - curve[:-1]
    dist_norm = gs.sqrt(gs.sum(gs.square(dist), axis=1))

    if gs.any(dist_norm < tol):
        for i in range(len(curve) - 1):
            if gs.sqrt(gs.sum(gs.square(curve[i + 1] - curve[i]), axis=0)) < tol:
                curve[i + 1] = (curve[i] + curve[i + 2]) / 2

    return curve


def load_treated_osteosarcoma_cells(n_sampling_points=0):
    """Load dataset of osteosarcoma cells (bone cancer cells).

    This cell dataset contains cell boundaries of mouse osteosarcoma
    (bone cancer) cells. The dlm8 cell line is derived from dunn and is more
    aggressive as a cancer. The cells have been treated with one of three
    treatments : control (no treatment), jasp (jasplakinolide)
    and cytd (cytochalasin D). These are drugs which perturb the cytoskelet
    of the cells.

    Parameters
    ----------
    n_sampling_points : int
        Number of points used to interpolate each cell boundary.
        Optional, Default: 0.
        If equal to 0, then no interpolation is performed.

    Returns
    -------
    cells : list of 650 planar discrete curves
        Each curve represents the boundary of a cell in counterclockwise order,
        their lengths are not necessarily equal.
    lines : list of 650 strings
        List of the cell lines of each cell (dlm8 or dunn).
    treatments : list of 650 strings
        List of the treatments given to each cell (control, cytd or jasp).
    """
    cells, lines, treatments = data_utils.load_cells()
    if n_sampling_points > 0:
        cells = _interpolate(cells, n_sampling_points)
    cells = _remove_consecutive_duplicates(cells)
    return cells, lines, treatments


def load_mutated_retinal_cells(n_sampling_points=0):
    """Load dataset of osteosarcoma cells (bone cancer cells).

    This cell dataset contains cell boundaries of mouse osteosarcoma
    (bone cancer) cells. The dlm8 cell line is derived from dunn and is more
    aggressive as a cancer. The cells have been treated with one of three
    treatments : control (no treatment), jasp (jasplakinolide)
    and cytd (cytochalasin D). These are drugs which perturb the cytoskelet
    of the cells.

    Parameters
    ----------
    n_sampling_points : int
        Number of points used to interpolate each cell boundary.
        Optional, Default: 0.
        If equal to 0, then no interpolation is performed.

    Returns
    -------
    cells : list of 650 planar discrete curves
        Each curve represents the boundary of a cell in counterclockwise order,
        their lengths are not necessarily equal.
    surfaces : list of 650 strings
        List of the cell lines of each cell (dlm8 or dunn).
    mutations : list of 650 strings
        List of the treatments given to each cell (control, cytd or jasp).
    """
    cells = open("mutated_retinal_cells/cells.txt", "r").read().split("\n\n")
    surfaces = open("mutated_retinal_cells/surfaces.txt", "r").read().split("\n")
    mutations = open("mutated_retinal_cells/mutations.txt", "r").read().split("\n")
    for i, cell in enumerate(cells):
        # command added to remove line break at the beginning of each cell
        # cell = cell[2:]
        cell = cell.split("\n")
        curve = []
        for point in cell:
            coords = [int(coord) for coord in point.split()]
            curve.append(coords)
        cells[i] = gs.cast(gs.array(curve), gs.float32)

    cells = cells[:-1]
    surfaces = surfaces[:-1]  # command added to remove space at the end
    mutations = mutations[:-1]  # command added to remove space at the end

    if n_sampling_points > 0:
        cells = _interpolate(cells, n_sampling_points)
    cells = _remove_consecutive_duplicates(cells)

    return cells, surfaces, mutations

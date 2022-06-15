"""Utils to load experimental datasets of cells."""


import geomstats.backend as gs
import geomstats.datasets.utils as data_utils
from geomstats.geometry.pre_shape import PreShapeSpace

M_AMBIENT = 2


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
    pos = gs.array(0.0, dtype=gs.float32)
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
    dist_norm = gs.sqrt(gs.sum(dist**2, axis=1))

    if gs.any(dist_norm < tol):
        for i in range(len(curve) - 1):
            if gs.sqrt(gs.sum((curve[i + 1] - curve[i]) ** 2, axis=0)) < tol:
                curve[i + 1] = (curve[i] + curve[i + 2]) / 2

    return curve


def _project_in_shape_space(curve, base_curve):
    """Project a curve in shape space.

    This happens in 2 steps:
    - remove translation (and scaling?) by projecting in pre-shape space.
    - remove rotation by exhaustive alignment minimizing the L² distance.

    Returns
    -------
    aligned_curve : discrete curve
    """
    n_sampling_points = curve.shape[-2]
    preshape = PreShapeSpace(m_ambient=M_AMBIENT, k_landmarks=n_sampling_points)
    curve = preshape.projection(curve)

    nb_sampling = len(curve)
    distances = gs.zeros(nb_sampling)
    for shift in range(nb_sampling):
        reparametrized = [curve[(i + shift) % nb_sampling] for i in range(nb_sampling)]
        aligned = preshape.align(point=reparametrized, base_point=base_curve)
        distances[shift] = preshape.norm(gs.array(aligned) - gs.array(base_curve))
    shift_min = gs.argmin(distances)
    reparametrized_min = [
        curve[(i + shift_min) % nb_sampling] for i in range(nb_sampling)
    ]
    aligned_curve = preshape.embedding_metric.align(
        point=reparametrized_min, base_point=base_curve
    )
    return aligned_curve


def load_treated_osteosarcoma_cells(n_sampling_points=10):
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
        print(f"Interpolating: All cells get {n_sampling_points} samplings points")
        for i_cell, cell in enumerate(cells):
            cells[i_cell] = _interpolate(cell, n_sampling_points)
        cells = gs.stack(cells, axis=0)

    print("Removing potential duplicate points.")
    for i_cell, cell in enumerate(cells):
        cells[i_cell] = _remove_consecutive_duplicates(cell)

    print("Projecting in shape space.")
    for i_cell, cell in enumerate(cells):
        cells[i_cell] = _project_in_shape_space(cell, cells[0])

    return cells, lines, treatments


def load_mutated_retinal_cells(n_sampling_points=10):
    """Load dataset of mutated retinal cells.

    The cells are grouped by mutation in the dataset :
    - the *control* cells are ARPE19,
    - the cells treated with Akt mutation,
    - and the ones treated with Mek mutation
    - and the cells treated with the Ras mutation.

    Additionally, in each of these classes, the cells are cultured on two surfaces :
    - the *GDA* cells (simple glass)
    - the *FN* ones (Fibronectin coated glass).

    Parameters
    ----------
    n_sampling_points : int
        Number of points used to interpolate each cell boundary.
        Optional, Default: 0.
        If equal to 0, then no interpolation is performed.

    Returns
    -------
    cells : list of 3871 planar discrete curves
        Each curve represents the boundary of a cell in counterclockwise order,
        their lengths are not necessarily equal.
    surfaces : list of 3871 strings
        List of the surfaces whre each cell has been cultivated.
    mutations : list of 3871 strings
        List of the mutations given to each cell .
    """
    cells = (
        open("cells/datasets/mutated_retinal_cells/cells.txt", "r").read().split("\n\n")
    )
    surfaces = (
        open("cells/datasets/mutated_retinal_cells/surfaces.txt", "r")
        .read()
        .split("\n")
    )
    mutations = (
        open("cells/datasets/mutated_retinal_cells/mutations.txt", "r")
        .read()
        .split("\n")
    )
    for i, cell in enumerate(cells):
        cell = cell.split("\n")
        curve = []
        for point in cell:
            coords = [int(coord) for coord in point.split()]
            curve.append(coords)
        cells[i] = gs.cast(gs.array(curve), gs.float32)

    if n_sampling_points > 0:
        print(f"Interpolating: All cells get {n_sampling_points} samplings points")
        for i_cell, cell in enumerate(cells):
            cells[i_cell] = _interpolate(cell, n_sampling_points)
        cells = gs.stack(cells, axis=0)

    print("Removing potential duplicate points.")
    for i_cell, cell in enumerate(cells):
        cells[i_cell] = _remove_consecutive_duplicates(cell)

    print("Projecting in shape space.")
    for i_cell, cell in enumerate(cells):
        cells[i_cell] = _project_in_shape_space(cell, cells[0])

    return cells, surfaces, mutations

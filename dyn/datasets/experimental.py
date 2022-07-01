"""Utils to load experimental datasets of cells."""

import glob
import os

import geomstats.backend as gs
import geomstats.datasets.utils as data_utils
import numpy as np
import skimage.io as skio
from geomstats.geometry.pre_shape import PreShapeSpace
from skimage import measure
from skimage.filters import threshold_otsu

import dyn.dyn.features.basic as basic

M_AMBIENT = 2


def _tiff_to_list(tiff_path):
    """Convert cell videos into trajectory of curves.

    Parameters
    ----------
    tiff_dir : absolute path of videos in .tif format.
    """
    img_stack = skio.imread(tiff_path, plugin="tifffile")
    cell_contours = []
    cell_imgs = []
    for img in img_stack:
        cell_imgs.append(img)
        thresh = threshold_otsu(img)
        binary = img > thresh
        contours = measure.find_contours(binary, 0.8)
        lengths = [len(c) for c in contours]
        max_length = max(lengths)
        index_max_length = lengths.index(max_length)
        cell_contours.append(contours[index_max_length])

    return cell_contours, cell_imgs


def _interpolate(curve, n_sampling_points):
    """Interpolate a discrete curve with nb_points from a discrete curve.

    Parameters
    ----------
    curve : array-like, shape=[n_points, 2]
    n_sampling_points : int

    Returns
    -------
    interpolation : array-like, shape=[n_sampling_points, 2]
       Discrete curve with n_sampling_points
    """
    old_length = curve.shape[0]
    interpolation = np.zeros((n_sampling_points, 2))
    incr = old_length / n_sampling_points
    pos = np.array(0.0, dtype=np.float32)
    for i in range(n_sampling_points):
        index = int(np.floor(pos))
        interpolation[i] = curve[index] + (pos - index) * (
            curve[(index + 1) % old_length] - curve[index]
        )
        pos += incr
    return gs.array(interpolation)


def _remove_consecutive_duplicates(curve, tol=1e-2):
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


def _exhaustive_align(curve, base_curve):
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

    nb_sampling = len(curve)
    distances = gs.zeros(nb_sampling)
    for shift in range(nb_sampling):
        reparametrized = gs.array(
            [curve[(i + shift) % nb_sampling] for i in range(nb_sampling)]
        )
        aligned = preshape.align(point=reparametrized, base_point=base_curve)
        distances[shift] = preshape.embedding_metric.norm(
            gs.array(aligned) - gs.array(base_curve)
        )
    shift_min = gs.argmin(distances)
    reparametrized_min = gs.array(
        [curve[(i + shift_min) % nb_sampling] for i in range(nb_sampling)]
    )
    aligned_curve = preshape.align(point=reparametrized_min, base_point=base_curve)
    return aligned_curve


def preprocess(cells, labels_a, labels_b, n_cells, n_sampling_points):
    """Preprocess a dataset of cells.

    Parameters
    ----------
    cells : list of all cells
        Each cell is an array of points in 2D.
    labels_a : list of str
        List of labels associated with each cell.
    labels_b : list of str
        List of labels associated with each cell.
    n_cells : int
        Number of cells to (randomly) select from this dataset.
    n_sampling_points : int
        Number of sampling points along the boundary of each cell.
    """
    if n_cells > 0:
        print(f"... Selecting only a random subset of {n_cells} / {len(cells)} cells.")
        indices = sorted(
            np.random.choice(gs.arange(0, len(cells), 1), size=n_cells, replace=False)
        )
        cells = [cells[idx] for idx in indices]
        labels_a = [labels_a[idx] for idx in indices]
        labels_b = [labels_b[idx] for idx in indices]

    if n_sampling_points > 0:
        print(
            "... Interpolating: "
            f"Cell boundaries have {n_sampling_points} samplings points."
        )
        for i_cell, cell in enumerate(cells):
            cells[i_cell] = _interpolate(cell, n_sampling_points)
        cells = gs.stack(cells, axis=0)

    print("... Removing potential duplicate sampling points on cell boundaries.")
    for i_cell, cell in enumerate(cells):
        cells[i_cell] = _remove_consecutive_duplicates(cell)

    print("\n- Cells and cell shapes: quotienting translation.")
    cells = cells - gs.mean(cells, axis=-2)[..., None, :]

    print("- Cell shapes: quotienting scaling (length).")
    cell_shapes = gs.zeros_like(cells)
    for i_cell, cell in enumerate(cells):
        cell_shapes[i_cell] = cell / basic.perimeter(cell)

    print("- Cell shapes: quotienting rotation.")
    for i_cell, cell_shape in enumerate(cell_shapes):
        cell_shapes[i_cell] = _exhaustive_align(cell_shape, cell_shapes[0])

    return cells, cell_shapes, labels_a, labels_b


def load_treated_osteosarcoma_cells(n_cells=-1, n_sampling_points=10):
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
    cells : array of n_cells planar discrete curves
        Each curve represents the boundary of a cell in counterclockwise order.
        Their barycenters are fixed at 0 (translation has been removed).
        Their lengths are not necessarily equal (scaling has not been removed).
    cell_shapes : array of n_cells planar discrete curves shapes
        Each curve represents the boundary of a cell in counterclockwise order.
        Their barycenters are fixed at 0 (translation has been removed).
        Their lengths are fixed at 1 (scaling has been removed).
        They are aligned in rotation to the first cell (rotation has been removed).
    lines : list of n_cells strings
        List of the cell lines of each cell (dlm8 or dunn).
    treatments : list of n_cells strings
        List of the treatments given to each cell (control, cytd or jasp).
    """
    cells, lines, treatments = data_utils.load_cells()
    return preprocess(cells, lines, treatments, n_cells, n_sampling_points)


def load_mutated_retinal_cells(n_cells=-1, n_sampling_points=10):
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
    cells : array of n_cells planar discrete curves
        Each curve represents the boundary of a cell in counterclockwise order.
        Their barycenters are fixed at 0 (translation has been removed).
        Their lengths are not necessarily equal (scaling has not been removed).
    cell_shapes : array of n_cells planar discrete curves shapes
        Each curve represents the boundary of a cell in counterclockwise order.
        Their barycenters are fixed at 0 (translation has been removed).
        Their lengths are fixed at 1 (scaling has been removed).
        They are aligned in rotation to the first cell (rotation has been removed).
    surfaces : list of n_cells strings
        List of the surfaces whre each cell has been cultivated.
    mutations : list of n_cells strings
        List of the mutations given to each cell .
    """
    cells = (
        open("dyn/datasets/mutated_retinal_cells/cells.txt", "r").read().split("\n\n")
    )
    surfaces = (
        open("dyn/datasets/mutated_retinal_cells/surfaces.txt", "r").read().split("\n")
    )
    mutations = (
        open("dyn/datasets/mutated_retinal_cells/mutations.txt", "r").read().split("\n")
    )

    for i, cell in enumerate(cells):
        cell = cell.split("\n")
        curve = []
        for point in cell:
            coords = [int(coord) for coord in point.split()]
            curve.append(coords)
        cells[i] = gs.cast(gs.array(curve), gs.float32)

    return preprocess(cells, surfaces, mutations, n_cells, n_sampling_points)


def load_trajectory_of_border_cells(n_sampling_points=10):
    """Load trajectories (or time-series) of border cells.

    Notes
    -----
    There are 25 images (frames) per .tif video.
    There are 16 videos (trajectories).

    Parameters
    ----------
    n_sampling_points : int
        Number of points sampled along the contour of a cell.

    Returns
    -------
    center_trajectories : array-like, shape=[16, 25, 2]
        2D coordinates of the barycenter of each cell's contours,
        for each of the 16 videos, for each of the 25 frames per video.
    shape_trajectories : array-like, shape=[16, 25, n_sampling_points, 2]
        2D coordinates of the sampling points defining the contour of each cell,
        for each of the 16 videos, for each of the 25 frames per video.
    labels : array-like, shape=[16,]
        Phenotype associated with each trajectory (video).
    """
    datasets_dir = os.path.dirname(os.path.realpath(__file__))
    list_tifs = glob.glob(
        os.path.join(datasets_dir, "single_border_protusion_cells/*.tif")
    )
    n_trajectories = len(list_tifs)
    one_img_stack = skio.imread(list_tifs[0], plugin="tifffile")
    n_time_points, height, width = one_img_stack.shape

    center_trajectories = gs.zeros((n_trajectories, n_time_points, 2))
    shape_trajectories = gs.zeros((n_trajectories, n_time_points, n_sampling_points, 2))
    img_trajectories = gs.zeros((n_trajectories, n_time_points, height, width))
    labels = []
    for i_traj, video_path in enumerate(list_tifs):
        video_name = os.path.basename(video_path)
        print(f"\n Processing trajectory {i_traj+1}/{n_trajectories}.")

        print(f"Converting {video_name} into list of cell contours...")
        cell_contours, cell_imgs = _tiff_to_list(video_path)

        labels.append(int(video_name.split("_")[0]))
        for i_contour, (contour, img) in enumerate(zip(cell_contours, cell_imgs)):
            interpolated = _interpolate(contour, n_sampling_points)
            cleaned = _remove_consecutive_duplicates(interpolated)
            center = gs.mean(cleaned, axis=-2)
            centered = cleaned - center[..., None, :]
            center_trajectories[i_traj, i_contour] = center
            shape_trajectories[i_traj, i_contour] = centered
            if img.shape != (height, width):
                print(
                    "Found image of a different size: "
                    f"{img.shape} instead of {height, width}. "
                    "Skipped image (not cell contours)."
                )
                continue
            img_trajectories[i_traj, i_contour] = gs.array(img.astype(float).T)
    labels = gs.array(labels)
    return center_trajectories, shape_trajectories, img_trajectories, labels

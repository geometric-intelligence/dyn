"""Utils to load experimental datasets of cells."""

import glob
import math as m
import os

# for septin cell alignment
import cv2
import geomstats.backend as gs
import geomstats.datasets.utils as data_utils
import numpy as np
import skimage.io as skio
from geomstats.geometry.pre_shape import PreShapeSpace
from skimage import measure
from skimage.filters import threshold_otsu

import dyn.dyn.features.basic as basic

M_AMBIENT = 2


def _tif_video_to_lists(tif_path):
    """Convert a cell video into two trajectories of contours and images.

    Parameters
    ----------
    tif_path : absolute path of video in .tif format.

    Returns
    -------
    contours_list : list of arrays
        List of 2D coordinates of points defining the contours of each cell
        within the video.
    imgs_list : list of array
        List of images in the input video.
    """
    img_stack = skio.imread(tif_path, plugin="tifffile")
    contours_list = []
    imgs_list = []
    for img in img_stack:
        imgs_list.append(img)
        thresh = threshold_otsu(img)
        binary = img > thresh
        contours = measure.find_contours(binary, 0.8)
        lengths = [len(c) for c in contours]
        max_length = max(lengths)
        index_max_length = lengths.index(max_length)
        contours_list.append(contours[index_max_length])

    return contours_list, imgs_list


def _septin_tif_video_to_lists(tif_path):
    """Convert a cell video into two trajectories of contours and images.

    special for septin because they are rgb images

    Parameters
    ----------
    tif_path : absolute path of video in .tif format.

    Returns
    -------
    contours_list : list of arrays
        List of 2D coordinates of points defining the contours of each cell
        within the video.
    imgs_list : list of array
        List of images in the input video.
    """
    img_stack_list = []
    for path in tif_path:
        img_stack_list.append(cv2.imread(tif_path[0], 0))
    img_stack = np.array(img_stack_list)
    contours_list = []
    imgs_list = []
    for img in img_stack:
        imgs_list.append(img)
        thresh = threshold_otsu(img)
        binary = img > thresh
        contours = measure.find_contours(binary, 0.8)
        lengths = [len(c) for c in contours]
        max_length = max(lengths)
        index_max_length = lengths.index(max_length)
        contours_list.append(contours[index_max_length])

    return contours_list, imgs_list


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
    return gs.array(interpolation, dtype=gs.float32)


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
    """Load trajectories (or time-series) of border cell clusters.

    The marker used for imaging is a GFP tagged marker of F-actin markers.

    Notes
    -----
    There are 25 images (frames) per .tif video.
    There are 16 videos (trajectories).

    In each movie, there is a group of cells called the "border cell cluster", i.e.
    a tightly packed group of 5-7 cells that coordinate their movement as they move.
    They look (and in many ways behave) as a single cell. They move within
    a large structure, the egg chamber: we say that they "migrate".

    At some point in their migration, the cluster shows has a "lead protrusion"
    and sometimes the cluster is compact and lack this protrusion.

    The movies from this dataset, i.e. labeled 33623, 59080, and 104438 are actually of
    single cells (one cell within the cluster) or sometimes two cells in the cluster
    labeled with the rest of the cells in the cluster being dark.

    The movies labeled "33623" represents the control case, and shows a single cell as
    the rest of the cells in the cluster are dark.

    The movies labeled "59080" and "104438" correspond to knockdowns of proteins,
    where we anticipate the following shape changes:
    - cells less cohesive,
    - producing very small thin protrusions.

    See Also
    --------
    - datasets/border_cell_cluster.png
    - datasets/border_cell_cluster.avi

    References
    ----------
    Campanale, Mondo, Montell (2022).
    Specialized protrusions coordinate migratory border cell cluster cohesion
    via Scribble, Cdep, and Rac.
    https://www.biorxiv.org/content/10.1101/2022.01.04.474957v1.full

    Parameters
    ----------
    n_sampling_points : int
        Number of points sampled along the contour of a cell.

    Returns
    -------
    centers_traj : array-like, shape=[16, 25, 2]
        2D coordinates of the barycenter of each cell's contours,
        for each of the 16 videos, for each of the 25 frames per video.
    shapes_traj : array-like, shape=[16, 25, n_sampling_points, 2]
        2D coordinates of the sampling points defining the contour of each cell,
        for each of the 16 videos, for each of the 25 frames per video.
    imgs_traj : array-like, shape=[16, 25, 512, 512]
        Images defining the videos, for each of the 16 videos.
    labels : array-like, shape=[16,]
        Phenotype associated with each trajectory (video).
    """
    datasets_dir = os.path.dirname(os.path.realpath(__file__))
    list_tifs = glob.glob(
        os.path.join(datasets_dir, "single_border_protusion_cells/*.tif")
    )

    n_traj = len(list_tifs)
    one_img_stack = skio.imread(list_tifs[0], plugin="tifffile")
    n_time_points, height, width = one_img_stack.shape

    centers_traj = gs.zeros((n_traj, n_time_points, 2))
    shapes_traj = gs.zeros((n_traj, n_time_points, n_sampling_points, 2))
    imgs_traj = gs.zeros((n_traj, n_time_points, height, width))
    labels = []
    for i_traj, video_path in enumerate(list_tifs):
        video_name = os.path.basename(video_path)
        print(f"\n Processing trajectory {i_traj+1}/{n_traj}.")

        print(f"Converting {video_name} into list of cell contours...")
        contours_list, imgs_list = _tif_video_to_lists(video_path)

        labels.append(int(video_name.split("_")[0]))
        for i_contour, (contour, img) in enumerate(zip(contours_list, imgs_list)):
            interpolated = _interpolate(contour, n_sampling_points)
            cleaned = _remove_consecutive_duplicates(interpolated)
            center = gs.mean(cleaned, axis=-2)
            centered = cleaned - center[..., None, :]
            centers_traj[i_traj, i_contour] = center
            shapes_traj[i_traj, i_contour] = centered
            if img.shape != (height, width):
                print(
                    "Found image of a different size: "
                    f"{img.shape} instead of {height, width}. "
                    "Skipped image (not cell contours)."
                )
                continue
            imgs_traj[i_traj, i_contour] = gs.array(img.astype(float).T)
    labels = gs.array(labels)
    return centers_traj, shapes_traj, imgs_traj, labels


# def _find_circle(tif_path):
#     """
#     takes a tif, returns the coordinates of the small circle that was placed
# on the septin cell files

#     the key function here is cv2.HoughCircles. But we needed very specific
# parameters in order to get
# the function
#     to detect our cirlces.
#     - minDist = 100 we knew that there was only one cirlce in the image, so
# we set this to be high so
# that there was no
#         way to get a false duplicate
#     - param1 = 100 we set this parameter to be high because "threshold value
# shough normally be
# higher, such as 300 or normally exposed and contrasty images."
#     - param2 = 10 we set this parameter to be low for detecting small circles
#     - minRadius =1, maxRadius = 10. We knew that our cirlces were only a few
# pixels wide (5), so we set
# these parameters acordingly
#     """

#     #print(tif_path)
#     img = skio.imread(tif_path, plugin="tifffile")
#     #img = cv2.imread(tif_path,0)

#     #print(np.nonzero(img))

#     #sigma = 3.0
#     #blurred_img = gaussian(img, sigma=(sigma, sigma), truncate=3.5)

#     #skview.ImageViewer( img)
#     #skview.show()


#     #plt.imshow(img)
#     #plt.show()

#     #io.imshow(img)
#     #plt.show()

#     #print(img)
#     #img = skio.imread(tif_path, plugin="tifffile")

#     #img = cv2.normalize(src=img, dst=None, alpha=0, beta=255,
# norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

#     #This is where the error is i think.
#     # detect circles in the image
#     #circle = np.empty([1,1,1])
#     circle = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp= 1,minDist=100,
# param1=100, param2=10, minRadius = 1, maxRadius=10 )
#     #cv2.HoughCircles(img, circle, cv2.HOUGH_GRADIENT, dp= int(3),minDist=0,
# param1=10, param2=60, minRadius = 0, maxRadius= -1)

#     #this returns "no circles found"
#     #circle = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp= 1, minDist = 1)

#     #circle = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10, np.array([]),
# 79, 23, 0, 100)
#     circle_array = np.array(circle)
#     #print(circle_array)
#     #print(len(circle))

#     # ensure at least some circles were found
#     #if not circle:
#     if not circle_array.any():
#         print("No circles found")
#     else:
#     #if circle is not None:
#         # convert the (x, y) coordinates and radius of the circles to integers
#         circle_array = np.round(circle_array).astype("int")
#     #else:
#         #print("No circles found")

#     print(circle_array[0][0])
#     return circle_array[0][0]


def _find_circle(tif_path):
    """Find a circle.

    Take a tif, returns the coordinates of the small circle that was placed on the
    septin cell files.

    The key function here is cv2.HoughCircles. But we needed very specific
    parameters in order to get the function to detect our cirlces.
    - minDist = 100 we knew that there was only one cirlce in the image, so we set this
        to be high so that there was no way to get a false duplicate
    - param1 = 100 we set this parameter to be high because "threshold value shough
    normally be higher, such as 300 or normally exposed and contrasty images."
    - param2 = 10 we set this parameter to be low for detecting small circles
    - minRadius =1, maxRadius = 10. We knew that our cirlces were only a few pixels wide
    (5), so we set these parameters acordingly
    """
    img = skio.imread(tif_path, plugin="tifffile")

    # this gives y first and then x. we will have to reverse.
    circle = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=100,
        param2=10,
        minRadius=1,
        maxRadius=10,
    )

    circle = np.array(circle)

    if not circle.any():
        print("No circles found")
    else:
        # reverses the order of x and y so that x comes first.
        x_circle = circle[0][0][1]
        y_circle = circle[0][0][0]
        circle = np.array([x_circle, y_circle])

    print(circle)
    return circle


def _determine_angle_sign(left_vector, circle_vector):
    """Find sign of angle to rotate cell about.

    This function tells us whether the angle that _septin_rotation_angle returns
    is a positive angle or a negative angle. We need this function because the
    arccos function that we use to find the magnitude of the angle only returns
    positive values, which means that the "_septin_align" fucntion will not know
    whether it needs to rotate the cell to the left or to the right in order to
    align the dot with the left side of the image frame.

    Thus, we use the cross product of the two vectors to determine whether the
    angle between left_vector and circle_vector is positive (i.e. the curve must be
    rotated counter clockwise in order to align the dot with the left edge) or
    negative (i.e. the curve must be rotated clockwise).

    If the z component of the cross product between these two vectors is negative,
    that means that the angle is positive.
    if the z component of the cross product between these two vectors is positive,
    that means that the angle is negative.

    Input vectors:
    --------------
    left_vector: 2D vector. tells us the location of the point that we are rotating
    the cell to
    circle_vector: 2D vector. tells us the location of the dot. i.e. the location
    of the part of
    the cell that we want to end up at the left side of the image.

    Returns:
    -------
    positive: tells whether the angle from the dot to the left point is positive or not.
    """
    threeD_left_vector = np.array([left_vector[0], left_vector[1], 0])
    threeD_circle_vector = np.array([circle_vector[0], circle_vector[1], 0])

    cross_product = np.cross(threeD_left_vector, threeD_circle_vector)

    if cross_product[2] > 0:
        positive = True
    else:
        positive = False

    return positive


def _hack_determine_angle_sign(left_vector, circle_vector):
    if left_vector[1] < circle_vector[1]:
        positive = True
    else:
        positive = False

    return positive


def _septin_rotation_angle(cell_center, tif_path):
    """Find rotation angle.

    This function aligns the curves so that they are pointing in the direction of
    motion.
    More specifically, each file is marked by a small dot. we are aligning the curves
    so that the small dot would fall on the left side of the picture frame.

    used tutorials:
    https://pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
    https://www.geeksforgeeks.org/how-to-detect-shapes-in-images-in-python-using-opencv/


    x_left and y_left are set at the left side of the image frame. the y_left
    coordiante is set at the same coordinate as the cell_center y coordinate so
    that we have an appropriate angle for which we can rotate the cell so that
    the "dotted" position is facing the left of the image.

    the picture frame is a square with side lengths 512, so the center of the left
    edge falls at (0,256)

    returns
    -------
    curve aligned so that the "direction of motion" is facing to the right.
    """
    # coordinates of the left middle of the image. determine this based on what
    # x and y circle are.
    x_left = 0
    y_left = cell_center[1]
    # print(cell_center)
    # print(y_left)

    # convert the (x, y) coordinates and radius of the circles
    # TO DO: DONT FORGET TO CHANGE BACK TO _FIND_CIRCLE AND EDIT THAT FUNCTION
    # x_circle,y_circle, r = _find_circle(tif_path)
    circle = _find_circle(tif_path)
    x_circle = circle[0]
    y_circle = circle[1]

    ############
    # testing. once we can get x,y coordinates of the circle, then we can use
    # stuff above.
    # cell_center = np.array([0.5,.5])

    # x_left = 0
    # y_left = 0.5

    # x_circle= 0.5
    # y_circle = 0
    #############

    # defining points
    left_point = np.array([x_left, y_left])
    circle_point = np.array([x_circle, y_circle])

    cell_center_tensor = cell_center
    cell_center = np.array(cell_center_tensor)

    # defining vector from center of curve to these points
    left_vector = left_point - cell_center
    circle_vector = circle_point - cell_center

    # unit vectors
    left_vector_u = left_vector / np.linalg.norm(left_vector)
    circle_vector_u = circle_vector / np.linalg.norm(circle_vector)

    positive = _hack_determine_angle_sign(left_vector, circle_vector)

    # now, find the angle between the two vectors

    # TDO: CHANGE VARIABLE POSITIVE TO BE "NEGATIVE"
    if positive:
        theta = np.arccos(np.clip(np.dot(left_vector_u, circle_vector_u), -1.0, 1.0))
    else:
        theta = -np.arccos(np.clip(np.dot(left_vector_u, circle_vector_u), -1.0, 1.0))

    return theta


def _septin_align(curve, theta):

    rotation = np.array([[m.cos(theta), -m.sin(theta)], [m.sin(theta), m.cos(theta)]])

    aligned_curve = curve @ rotation.T

    return aligned_curve


# def draft_load_septin_cells(group, n_sampling_points):
#     """ Load dataset of septin control cells.

#     There are three groups that we are considering: control, Septin Knockdown,
# Septin Overexpression.

#     Notes
#     -----
#     There are 36 tif files in Control -> binary files
#     There are 45 tif files in Septin Knockdown -> binary files
#     There are 36 tif files in Septin Overexpression -> binary files

#     current problem: i think that the algorithm does not know whether to rotate
# left or whether to rotate right (to get
#     the dots aligned)

#     actually, also instead of aligning the dots to the middle of the frame, we
# should be aligning them to the y coordinate of the cell center.
#     """
#     dataset_dir = os.path.dirname(os.path.realpath(__file__))

#     # os.path.join finds the path that leads you to the file
#     # glob.glob finds and returns the file you are looking for and returns the data.
#     #group_path = os.path.join(dataset_dir, "septin_groups/"+group+"/
# binary_images/*.tif")
#     group_path = os.path.join(dataset_dir, "septin_groups/"+group+"/
# dotted_binary_images/*.tif")
#     #align_septin_cell(group_path)
#     group_tifs = glob.glob(group_path)
#     print('Loading '+group+' data')
#     print('n_sampling_points= '+str(n_sampling_points))


#     #test_paths = os.path.join(dataset_dir, "septin_groups/"+group+"/
# dotted_oriented_images/BINARY_TEST.tif")
#     #test_tifs = glob.glob(test_paths)

#     #test_tifs_array = np.array(test_tifs)
#     #print(test_tifs_array.shape)

#     #img_stack_test = skio.imread(test_tifs, plugin="tifffile")
#     #print(img_stack_test.shape)

#     #this showed same shape as non-draft version, so this is not problem
#     #group_tifs_array = np.array(group_tifs)
#     #print(group_tifs_array.shape)


#     #before, was not working because new tifs are not grayscale
#     #img_stack_list = []
#     #for path in group_tifs:
#     #    img_stack_list.append(cv2.imread(group_tifs[0],0))
#     #img_stack = np.array(img_stack_list)
#     #print(img_stack.shape)

#     img_stack = skio.imread(group_tifs, plugin="tifffile")
#     n_images, height, width = img_stack.shape
#     print(img_stack.shape)

#     cell_centers = gs.zeros((n_images, 2))
#     cell_shapes = gs.zeros((n_images, n_sampling_points, 2))
#     cell_imgs = gs.zeros((n_images, height, width))


#     # This converts all the images into a list of contours and images.
#     contours_list, imgs_list = _tif_video_to_lists(group_tifs)
#     group_labels=[]
#     theta = []

#     for i_contour, (contour, img) in enumerate(zip(contours_list, imgs_list)):
#         interpolated = _interpolate(contour, n_sampling_points)
#         cleaned = _remove_consecutive_duplicates(interpolated)
#         center = gs.mean(cleaned, axis=-2)
#         centered = cleaned - center[..., None, :]
#         cell_centers[i_contour] = center
#         cell_shapes[i_contour] = centered
#         if img.shape != (height, width):
#             print(
#                 "Found image of a different size: "
#                 f"{img.shape} instead of {height, width}. "
#                 "Skipped image (not cell contours)."
#             )
#             continue
#         cell_imgs[i_contour] = gs.array(img.astype(float).T)
#         group_labels.append(group)

#         theta.append(_septin_rotation_angle(center,group_tifs[i_contour]))

#         #putting this here just for testing
#         #_find_circle(group_tifs[i_contour])

#         #this would be the center of that original image, plus the path to that
# image.
#         #print(septin_rotation_angle(center,group_tifs[i_contour]))
#         #theta.append(septin_rotation_angle(center,group_tifs[i_contour]))
#     theta_array = np.array(theta)

#     print("- Cell shapes: quotienting scaling (length).")
#     for i_cell, cell in enumerate(cell_shapes):
#         cell_shapes[i_cell] = cell / basic.perimeter(cell_shapes[i_cell])

#     print("- Cell shapes: properly aligning in direction of motion.")

#     for i_cell, cell_shape in enumerate(cell_shapes):
#         #change this line and replace it with something that aligns according to dot.
#         print("theta "+str(i_cell)+" : "+str(theta[i_cell]))
#         cell_shapes[i_cell] = _septin_align(cell_shape, theta[i_cell])

#     return cell_centers, cell_shapes, cell_imgs, group_labels


def load_septin_cells(group, n_sampling_points):
    """Load dataset of septin control cells.

    There are three groups that we are considering: control, Septin Knockdown,
    Septin Overexpression.

    Notes
    -----
    There are 36 tif files in Control -> binary files
    There are 45 tif files in Septin Knockdown -> binary files
    There are 36 tif files in Septin Overexpression -> binary files

    current problem: i think that the algorithm does not know whether to rotate
    left or whether to rotate right (to get the dots aligned)

    actually, also instead of aligning the dots to the middle of the frame, we
    should be aligning them to the y coordinate of the cell center.
    """
    dataset_dir = os.path.dirname(os.path.realpath(__file__))

    group_path = os.path.join(
        dataset_dir, "septin_groups/" + group + "/dotted_binary_images/*.tif"
    )
    group_tifs = glob.glob(group_path)
    print("Loading " + group + " data")
    print("n_sampling_points= " + str(n_sampling_points))

    img_stack = skio.imread(group_tifs, plugin="tifffile")
    n_images, height, width = img_stack.shape
    print(img_stack.shape)

    cell_centers = gs.zeros((n_images, 2))
    cell_shapes = gs.zeros((n_images, n_sampling_points, 2))
    cell_imgs = gs.zeros((n_images, height, width))

    # This converts all the images into a list of contours and images.
    contours_list, imgs_list = _tif_video_to_lists(group_tifs)
    group_labels = []
    theta = []
    circle_coords = []
    lefts = []

    for i_contour, (contour, img) in enumerate(zip(contours_list, imgs_list)):
        interpolated = _interpolate(contour, n_sampling_points)
        cleaned = _remove_consecutive_duplicates(interpolated)
        center = gs.mean(cleaned, axis=-2)
        centered = cleaned - center[..., None, :]
        cell_centers[i_contour] = center
        cell_shapes[i_contour] = centered
        if img.shape != (height, width):
            print(
                "Found image of a different size: "
                f"{img.shape} instead of {height, width}. "
                "Skipped image (not cell contours)."
            )
            continue
        cell_imgs[i_contour] = gs.array(img.astype(float).T)
        group_labels.append(group)

        circle_coords.append(_find_circle(group_tifs[i_contour]))
        theta.append(_septin_rotation_angle(center, group_tifs[i_contour]))
        lefts.append([0, center[1]])

        # putting this here just for testing
        # _find_circle(group_tifs[i_contour])

        # this would be the center of that original image, plus the path to that image.
        # print(septin_rotation_angle(center,group_tifs[i_contour]))
        # theta.append(septin_rotation_angle(center,group_tifs[i_contour]))
    theta_array = np.array(theta)
    circle_coords_array = np.array(circle_coords)
    lefts_array = np.array(lefts)

    print("- Cell shapes: quotienting scaling (length).")
    for i_cell, cell in enumerate(cell_shapes):
        cell_shapes[i_cell] = cell / basic.perimeter(cell_shapes[i_cell])

    print("- Cell shapes: properly aligning in direction of motion.")

    for i_cell, cell_shape in enumerate(cell_shapes):
        print("theta " + str(i_cell) + " : " + str(theta[i_cell]))
        cell_shapes[i_cell] = _septin_align(cell_shape, theta[i_cell])

    return (
        cell_centers,
        cell_shapes,
        cell_imgs,
        group_labels,
        theta_array,
        circle_coords_array,
        lefts_array,
        group_tifs,
    )


# def load_septin_cells(group, n_sampling_points):
#     """ Load dataset of septin control cells.

#     There are three groups that we are considering: control, Septin Knockdown,
# Septin Overexpression.

#     Notes
#     -----
#     There are 36 tif files in Control -> binary files
#     There are 45 tif files in Septin Knockdown -> binary files
#     There are 36 tif files in Septin Overexpression -> binary files
#     """
#     dataset_dir = os.path.dirname(os.path.realpath(__file__))

#     # os.path.join finds the path that leads you to the file
#     # glob.glob finds and returns the file you are looking for and returns
# the data.
#     group_path = os.path.join(dataset_dir, "septin_groups/"+group+"/
# binary_images/*.tif")
#     group_tifs = glob.glob(group_path)
#     print('Loading '+group+' data')
#     print('n_sampling_points= '+str(n_sampling_points))

#     img_stack = skio.imread(group_tifs, plugin="tifffile")
#     n_images, height, width = img_stack.shape

#     cell_centers = gs.zeros((n_images, 2))
#     cell_shapes = gs.zeros((n_images, n_sampling_points, 2))
#     cell_imgs = gs.zeros((n_images, height, width))


#     # This converts all the images into a list of contours and images.
#     contours_list, imgs_list = _tif_video_to_lists(group_tifs)
#     group_labels=[]

#     for i_contour, (contour, img) in enumerate(zip(contours_list, imgs_list)):
#         interpolated = _interpolate(contour, n_sampling_points)
#         cleaned = _remove_consecutive_duplicates(interpolated)
#         center = gs.mean(cleaned, axis=-2)
#         centered = cleaned - center[..., None, :]
#         cell_centers[i_contour] = center
#         cell_shapes[i_contour] = centered
#         if img.shape != (height, width):
#             print(
#                 "Found image of a different size: "
#                 f"{img.shape} instead of {height, width}. "
#                 "Skipped image (not cell contours)."
#             )
#             continue
#         cell_imgs[i_contour] = gs.array(img.astype(float).T)
#         group_labels.append(group)

#     print("- Cell shapes: quotienting scaling (length).")
#     for i_cell, cell in enumerate(cell_shapes):
#         cell_shapes[i_cell] = cell / basic.perimeter(cell_shapes[i_cell])

#     print("- Cell shapes: quotienting rotation.")
#     for i_cell, cell_shape in enumerate(cell_shapes):
#         #change this line and replace it with something that aligns according
# to dot.
#         #might actually want to do this before the cell is centered. align the
# cell and then
#         #center it so that there are no issues with having to re-center the dot.
#         cell_shapes[i_cell] = _exhaustive_align(cell_shape, cell_shapes[0])

#     return cell_centers, cell_shapes, cell_imgs, group_labels

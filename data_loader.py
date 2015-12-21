from os import listdir
from os.path import join, isdir
from sklearn.datasets.base import Bunch

import logging
import numpy as np

logger = logging.getLogger()

data_folder_path = "/Users/Dongbo/Documents/Education/NYU/Courses/2015 Fall/Foundations Of Machine Learning/Project/project/FaceRecognition/managed_data"

#
# Scale the faces to relative color
#
def scaleFaces(faces):

    scaledFaces = faces - faces.min()
    scaledFaces /= scaledFaces.max()
    scaledFaces *= 255

    return scaledFaces


#
# Load images and scale uint8 coded colors to the [0.0, 1.0] floats
#
def loadImages(file_paths, slice_, color, resize):
    try:
        try:
            from scipy.misc import imread
        except ImportError:
            from scipy.misc.pilutil import imread
        from scipy.misc import imresize
    except ImportError:
        raise ImportError("The Python Imaging Library (PIL)"
                          " is required to load data from jpeg files")

    # compute the portion of the images to load to respect the slice_ parameter
    # given by the caller
    default_slice = (slice(0, 188), slice(0, 250))
    if slice_ is None:
        slice_ = default_slice
    else:
        slice_ = tuple(s or ds for s, ds in zip(slice_, default_slice))

    h_slice, w_slice = slice_
    h = (h_slice.stop - h_slice.start) // (h_slice.step or 1)
    w = (w_slice.stop - w_slice.start) // (w_slice.step or 1)

    if resize is not None:
        resize = float(resize)
        h = int(resize * h)
        w = int(resize * w)

    # allocate some contiguous memory to host the decoded image slices
    n_faces = len(file_paths)
    if not color:
        faces = np.zeros((n_faces, h, w), dtype=np.float32)
    else:
        faces = np.zeros((n_faces, h, w, 3), dtype=np.float32)

    # iterate over the collected file path to load the jpeg files as numpy
    # arrays
    for i, file_path in enumerate(file_paths):
        if i % 1000 == 0:
            logger.info("Loading face #%05d / %05d", i + 1, n_faces)

        # Checks if jpeg reading worked. Refer to issue #3594 for more
        # details.

        img = imread(file_path)
        if img.ndim is 0:
            raise RuntimeError("Failed to read the image file %s, "
                               "Please make sure that libjpeg is installed"
                               % file_path)

        face = np.asarray(img[slice_], dtype=np.float32)

        if resize is not None:
            face = imresize(face, resize)
        if not color:

            # average the color channels to compute a gray levels
            # representaion
            face = face.mean(axis=2)

        faces[i, ...] = face

        scaleFaces(faces)

    return faces

#
# Load Face by picture with names
#
def load_data(data_folder_path, slice_=None, color=False, resize=None,
              min_faces_per_person=0):

    # scan the data folder content to retain people with more that
    # `min_faces_per_person` face pictures
    person_names, file_paths = [], []

    for person_name in sorted(listdir(data_folder_path)):

        folder_path = join(data_folder_path, person_name)
        if not isdir(folder_path):
            continue
        paths = []
        for f in listdir(folder_path):
            if f != '.DS_Store':
                paths.append(join(folder_path, f))

        n_pictures = len(paths)

        if n_pictures >= min_faces_per_person:
            person_name = person_name.replace('_', ' ')

            person_names.extend([person_name] * n_pictures)

            file_paths.extend(paths)

    n_faces = len(file_paths)

    if n_faces == 0:
        raise ValueError("min_faces_per_person=%d is too restrictive" %
                         min_faces_per_person)

    target_names = np.unique(person_names)
    target = np.searchsorted(target_names, person_names)

    faces = loadImages(file_paths, slice_, color, resize)

    # shuffle the faces with a deterministic RNG scheme to avoid having
    # all faces of the same person in a row, as it would break some
    # cross validation and learning algorithms such as SGD and online
    # k-means that make an IID assumption

    indices = np.arange(n_faces)
    np.random.RandomState(42).shuffle(indices)
    faces, target = faces[indices], target[indices]

    return faces, target, target_names


#
# bunch needed date for dimention reductor and classifier use
#
def getDataBunch(data_home=None, funneled=True, resize=0.5,
                 min_faces_per_person=0, color=False,
                 slice_=(slice(45, 205), slice(14, 174))):
    logger.info('Loading people faces from %s', data_folder_path)

    # load and memoize the pairs as np arrays
    faces, target, target_names = load_data(
            data_folder_path, resize=resize,
            min_faces_per_person=min_faces_per_person, color=color, slice_=slice_)

    # pack the results as a Bunch instance
    return Bunch(data=faces.reshape(len(faces), -1), images=faces,
                 target=target, target_names=target_names,
                 DESCR="LFW faces dataset")

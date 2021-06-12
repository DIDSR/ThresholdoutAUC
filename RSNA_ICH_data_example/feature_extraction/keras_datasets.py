import os

import cv2
import numpy as np
from skimage.transform import resize
from tensorflow.keras.utils import Sequence


class TestDataGenerator(Sequence):
    """Data Generator for Keras

    Sequence based data generator. Suitable for building data generator for
    extraction of features from images using different deep
    learning models trained on ImageNet.

    Parameters
    ----------
    img_dir : str or None
        Path to the image directory. At least one of `img_dir` or
        `img_files` needs to be provided (i.e., to be not None). If
        `img_files` is None then it will be attempted to load all
        data contained in `img_dir`.
    img_files : list or None
        A list of paths to individual image files, which are expected to
        be 2D images in formats supported by OpenCV. If `img_dir` is also
        provided then `img_files` refers to file names in that directory.
        If `img_dir` is not provided then `img_files` needs to include the
        full paths to the images.
    batch_size : int
        Batch size for the CNN
    image_size : tuple
        Size to which the images will be resized, a tuple of 2 int.
    preprocess_input : function
        Expects one of the `preprocess_input` functions from
        `tensorflow.keras.applications...`

    Attributes
    ----------
    img_files : list or None
    img_dir : str or None
    batch_size : int
    image_size : tuple
    preprocess_input : function
    idx : numpy.ndarray
    """
    def __init__(self, img_dir=None, img_files=None, batch_size=32,
                 image_size=(224, 224), preprocess_input=None):
        assert img_dir is not None or img_files is not None, \
            "Provide input images either as 'img_dir' or as 'img_files'."

        if img_dir is not None:
            if img_files is None:
                img_files = os.listdir(img_dir)
            self.img_files = [os.path.join(img_dir, f) for f in img_files]
        else:
            self.img_files = img_files
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.preprocess_input = preprocess_input
        self.idx = np.arange(len(self.img_files))

    def __len__(self):
        """Returns the number of batches per epoch

        """
        return int(np.ceil(len(self.img_files) / self.batch_size))

    def __getitem__(self, index):
        """ Generate one batch of data

        Parameters
        ----------
        index : int
            Indices of the images for the batch

        Returns
        -------
        x : numpy.ndarray
            One batch of images
        """
        idx = self.idx[(index * self.batch_size):((index + 1) * self.batch_size)]
        img_files_batch = [self.img_files[k] for k in idx]
        x = self._generate_batch(img_files_batch)
        return x

    def _generate_batch(self, img_files_batch):
        """Generates data containing batch_size images

        Parameters
        ----------
        img_files_batch : list
            Images to load; a list of paths to the images.

        Returns
        -------
        x : numpy.ndarray
            One batch of images.
        """

        img_list = [None for _ in img_files_batch]
        for i, img_file in enumerate(img_files_batch):
            img_list[i] = cv2.imread(img_file)

        if self.img_dir is not None:
            assert all([img is not None for img in img_list]), \
                "Remove files/images of unsupported file formats from {}.".format(self.img_dir)
        else:
            assert all([img is not None for img in img_list]), \
                "Either files/images of unsupported file formats were provided, or maybe you did not provide the full file paths?"

        num_slices = len(img_list)
        x = np.zeros((num_slices, *self.image_size, 3))

        for idx, img_slice in enumerate(img_list):
            # resize and preprocess to be compatible with ImageNet trained network
            resized_slice = resize(
                img_slice, self.image_size, anti_aliasing=True
            )
            resized_slice = self.preprocess_input(resized_slice)
            if resized_slice.ndim == 2:
                # create an "RGB" image by repeating the slice 3 times
                x[idx, :, :, 0] = resized_slice
                x[idx, :, :, 1] = resized_slice
                x[idx, :, :, 2] = resized_slice
            elif resized_slice.ndim == 3:
                x[idx, :, :, :] = resized_slice
            else:
                raise Exception("Image shape must be in the form (height, width) or (height, width, 3).")

        return x

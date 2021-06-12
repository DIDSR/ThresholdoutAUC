import cv2
import numpy as np
from skimage.transform import resize
import tensorflow.keras.models

from keras_datasets import TestDataGenerator


class CNNFeatureExtractor:
    """ Feature extractor using convolutional neural networks

    Extract features from a DICOM medical image using
    different deep learning models trained on ImageNet.

    Parameters
    ----------
    model : str
        Name of the pre-trained deep learning model used for feature
        extraction. Can be one of "ResNet50", "DenseNet121", "DenseNet201",
        "VGG16", "VGG19", "MobileNet", "MobileNetV2", "Xception", "InceptionV3"

    Attributes
    ----------
    feature_extractor
        A convolutional neural network pre-trained on ImageNet with the last
        layer(s) removed. The pre-trained networks are loaded from
        `tensorflow.keras.applications` before being modified to remove the
        last fully connected laser(s).
    image_size : tuple
        A tuple of two integers specifying the height and width dimensions
        of the input image. Needs to be compatible with the `model`, i.e.,
        the pre-trained CNN of choice.
    preprocess_input : function
        A function to preprocess (rescale, etc.) the input images to be
        compatible with the pre-trained CNN of choice. Likely one of the
        `preprocess_input` functions from `tensorflow.keras.applications`
    """
    def __init__(self, model="ResNet50"):
        self.model = model
        # define model
        m = self._get_ImageNet_trained_model(model)
        self.feature_extractor, self.image_size, self.preprocess_input = m

    def get_features(
            self, img_dir=None, img_files=None, batch_size=1,
            verbose=1, workers=1, use_multiprocessing=False
    ):
        """Extract features

        Extract features from medical images in DICOM formats or formats
        supported by OpenCV using different convolutional neural network
        models trained on ImageNet.

        Parameters
        ----------
        img_dir : str or None
            Path to the image directory. At least one of `img_dir` or
            `img_files` needs to be provided (i.e., to be not None). If
            `img_files` is None then it will be attempted to load all
            data contained in `img_dir`.
        img_files : list or None
            A list of paths to individual image files, which are expected to
            be images in formats supported by OpenCV. If `img_dir` is also
            provided then `img_files` refers to file names in that directory.
            If `img_dir` is not provided then `img_files` needs to include
            the full paths to the images.
        batch_size : int
            Batch size for the CNN.
        verbose : int
            Level of verbosity, 0 or 1. Defaults to 1.
        workers : int
            Used for the data loader `imgproc.keras_datasets.TestDataGenerator`
            only.  Maximum number of processes to spin up when using
            process-based threading.  If unspecified, workers will default to
            1. If 0, will execute the data loader on the main thread.
        use_multiprocessing : bool
            Used for the data loader `imgproc.keras_datasets.TestDataGenerator`
            only.  If True, use process-based threading. If unspecified,
            `use_multiprocessing` will default to False.

        Returns
        -------
        features : numpy.ndarray
            The extracted features.
        """
        data_gen = TestDataGenerator(
            img_dir,
            img_files,
            batch_size,
            self.image_size,
            self.preprocess_input
        )
        features = self.feature_extractor.predict(
            data_gen,
            verbose=verbose,
            workers=workers,
            use_multiprocessing=use_multiprocessing
        )
        return features, data_gen.img_files

    def _get_ImageNet_trained_model(self, model):
        """Initialize an ImageNet pre-trained model

        After the chosen convolution neural network (see `model` for options)
        is loaded with the ImageNet pre-trained weights from
        `tensorflow.keras.applications`, the last fully connected layer(s) is
        removed to turn the classification model into a feature extractor.

        Parameters
        ----------
        model : str
            Name of the pre-trained deep learning model to be used for feature
            extraction. Can be one of "ResNet50", "DenseNet121", "DenseNet201",
            "VGG16", "VGG19", "MobileNet", "MobileNetV2", "Xception",
            "InceptionV3".

        Returns
        -------
        feature_extractor
            A Keras model object.
        image_size : tuple
            Image size as (height, width).
        preprocess_input
            A preprocessing function - the `preprocess_input` function from
            `tensorflow.keras.applications` that corresponds to the chosen `model`.
        """
        if model == "ResNet50":
            from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

            image_size = (224, 224)
            res = ResNet50(
                input_shape=image_size + (3,), weights="imagenet", include_top=True
            )
            feature_extractor = tensorflow.keras.models.Model(
                inputs=res.input, outputs=res.get_layer("avg_pool").output
            )
        elif model == "DenseNet121":
            from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input

            image_size = (224, 224)
            dense121 = DenseNet121(
                input_shape=image_size + (3,), weights="imagenet", include_top=True
            )
            feature_extractor = tensorflow.keras.models.Model(
                inputs=dense121.input, outputs=dense121.get_layer("avg_pool").output
            )
        elif model == "DenseNet201":
            from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input

            image_size = (224, 224)
            dense201 = DenseNet201(
                input_shape=image_size + (3,), weights="imagenet", include_top=True
            )
            feature_extractor = tensorflow.keras.models.Model(
                inputs=dense201.input, outputs=dense201.get_layer("avg_pool").output
            )
        elif model == "VGG16":
            from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

            image_size = (224, 224)
            vgg16 = VGG16(
                input_shape=image_size + (3,), weights="imagenet", include_top=True
            )
            feature_extractor = tensorflow.keras.models.Model(
                inputs=vgg16.input, outputs=vgg16.get_layer("fc2").output
            )
        elif model == "VGG19":
            from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input

            image_size = (224, 224)
            vgg19 = VGG19(
                input_shape=image_size + (3,), weights="imagenet", include_top=True
            )
            feature_extractor = tensorflow.keras.models.Model(
                inputs=vgg19.input, outputs=vgg19.get_layer("fc2").output
            )
        elif model == "MobileNet":
            from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input

            image_size = (224, 224)
            mobile = MobileNet(
                input_shape=image_size + (3,), weights="imagenet", include_top=True
            )
            feature_extractor = tensorflow.keras.models.Model(
                inputs=mobile.input, outputs=mobile.get_layer(index=-6).output
            )
            # the output layer in the above is 'global_average_pooling2d', but we use the equivalent 'index' due to a bug in tensorflow
        elif model == "MobileNetV2":
            from tensorflow.keras.applications.mobilenet_v2 import (
                MobileNetV2,
                preprocess_input,
            )

            image_size = (224, 224)
            mobilev2 = MobileNetV2(
                input_shape=image_size + (3,), weights="imagenet", include_top=True
            )
            feature_extractor = tensorflow.keras.models.Model(
                inputs=mobilev2.input, outputs=mobilev2.get_layer(index=-2).output
            )
            # output layer in the above is 'global_average_pooling2d', but we use the equivalent 'index' due to a bug in tensorflow
        elif model == "Xception":
            from tensorflow.keras.applications.xception import Xception, preprocess_input

            image_size = (299, 299)
            xception = Xception(
                input_shape=image_size + (3,), weights="imagenet", include_top=True
            )
            feature_extractor = tensorflow.keras.models.Model(
                inputs=xception.input, outputs=xception.get_layer("avg_pool").output
            )
        elif model == "InceptionV3":
            from tensorflow.keras.applications.inception_v3 import (
                InceptionV3,
                preprocess_input,
            )

            image_size = (299, 299)
            inception = InceptionV3(
                input_shape=image_size + (3,), weights="imagenet", include_top=True
            )
            feature_extractor = tensorflow.keras.models.Model(
                inputs=inception.input, outputs=inception.get_layer("avg_pool").output
            )
        else:
            raise NotImplementedError("please specify a model!")

        return feature_extractor, image_size, preprocess_input

#!/usr/bin/env python
# coding: utf-8

import os

import cv2
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

# Global variables:

DATA_BASE_PATH = '../data/'  # location of raw data
TRAIN_DIR = 'stage_2_train_images/'
TEST_DIR = 'stage_2_test_images/'
PNG_DIR = 'png_256/'  # where to save preprocessed data
TRAIN_IMG_STATS_FILE = 'train_img_stats.csv'  # where to write meta data and some pixel statistics for each train image
TEST_IMG_STATS_FILE = 'test_img_stats.csv'  # where to write meta data and some pixel statistics for each test image
RESIZE = 256  # crop and resize images to a square of this size
MINCROP = 256  # don't crop so much so that the resulting image is less than this size


# Auxilliary functions

def get_first_of_dicom_field(x):
    """
    Get x[0] if x is a 'pydicom.multival.MultiValue', otherwise get x.
    Each result is transformed into an Int if possible, otherwise transformed to a String.
    """
    if isinstance(x, pydicom.multival.MultiValue):
        result = x[0]
    else:
        result = x
    # transform to int or str
    try:
        result = int(result)
    except (ValueError, TypeError):
        result = str(result)
    return result


def get_dicom_fields(data):
    """
    Get slope, intercept, windowing parameters, etc. from the DICOM header
    """
    dicom_field_names = [
        "patient_ID",
        "study_instance_ID",
        "series_instance_ID",
        "study_ID",
        "bits_allocated",
        "bits_stored",
        "pixel_representation",
        "window_center",
        "window_width",
        "intercept",
        "slope"]
    dicom_fields_raw = [
        data[('0010', '0020')].value,  # patient ID
        data[('0020', '000D')].value,  # study instance UID
        data[('0020', '000E')].value,  # series instance UID
        data[('0020', '0010')].value,  # study ID
        data[('0028', '0100')].value,  # bits allocated
        data[('0028', '0101')].value,  # bits stored
        data[('0028', '0103')].value,  # pixel representation (0: unsigned int, 1: signed int)
        data[('0028', '1050')].value,  # window center
        data[('0028', '1051')].value,  # window width
        data[('0028', '1052')].value,  # intercept
        data[('0028', '1053')].value   # slope
    ]
    dicom_fields_values = [get_first_of_dicom_field(x) for x in dicom_fields_raw]
    return dict(zip(dicom_field_names, dicom_fields_values))


def make_12bit(img):
    """
    Note: this function is not needed for the given data.

    Transform the input image form 16 bit into 12 bit format.
    Assumes that the pixel values have been transformed to
    floating point numbers.
    """
    assert img.dtype == "float32"
    if np.max(img) - np.min(img) > 4096:
        img = img * 4096.0 / 65536.0
    return img


def crop_and_square(img):
    """Crop background, then pad the resulting image to be square"""
    assert np.all(img >= 0.0), "the input image cannot have pixel values greater than 0."
    if np.all(img == 0.0):
        # i.e., the image is empty
        cropped_img = img
    else:
        # crop some of the background
        nonzero = np.where(img > 0.0)
        min_x = np.min(nonzero[0])
        max_x = np.max(nonzero[0])
        min_y = np.min(nonzero[1])
        max_y = np.max(nonzero[1])
        cropped_img = img[min_x:(max_x+1), min_y:(max_y+1), :]
    # pad to square of size at least 224-by-224
    h, w, _ = cropped_img.shape
    if h != w or max(h, w) < MINCROP:
        m = max(h, w)
        new_size = max(m, MINCROP)
        min_value = 0.0  # np.min(cropped_img) # see the assert above
        padded_img = np.full((new_size, new_size, cropped_img.shape[-1]), min_value)
        rh = (m-h)//2
        rw = (m-w)//2
        padded_img[rh:(rh+h), rw:(rw+w), :] = cropped_img
    else:
        padded_img = cropped_img
    # sanity checks
    assert padded_img.shape[0] >= MINCROP or img.shape[0] < MINCROP, "cropped&padded shape is {} but image shape is {}".format(padded_img.shape, img.shape)
    assert padded_img.shape[1] >= MINCROP or img.shape[1] < MINCROP, "cropped&padded shape is {} but image shape is {}".format(padded_img.shape, img.shape)
    return padded_img


def apply_slope_intercept_rescale_make_channels(img, dicom_fields, resize_dim, verbose=False):
    """
    Creates an output image with three different channels,
    corresponding to different windowing of the dicom; notes:
    * some input images are of unsigned int type, so we transform
      everything to float before processing;
    * some input images are stored in 16 bit format and some  as 12 bit;
      we transform all images to 16 bit.
    """
    out_img = np.zeros((*resize_dim, 3), dtype="float32")
    img = np.array(img, dtype="float32")
    # adjustment based on a comment by Malcolm McLean to the kaggle kernel some-dicom-gotchas-to-be-aware-of-fastai by Jeremy Howard
    if dicom_fields["pixel_representation"] == 0 and dicom_fields["intercept"] > -2:
        if verbose:
            print("Suspicious image:")
            print(dicom_fields)
            print(np.percentile(img, np.linspace(1, 100, 19)))
        img[img > 3093.0] -= 4097.0
    # apply slope and intercept
    shifted_img = img*dicom_fields["slope"] + dicom_fields["intercept"]
    # keep track of some statistics; these should be consistent across all images now
    pixel_stats = np.percentile(shifted_img, np.linspace(1, 100, 19))

    # first channel: keep HU range 0-255, includes the standard brain window
    channel0 = shifted_img.copy()
    img_min = 0.0
    img_max = 255.0
    channel0[channel0 < img_min] = img_min
    channel0[channel0 > img_max] = img_max

    # second channel (subdural window)
    channel1 = shifted_img.copy()
    window_center = 100.0
    window_width = 255.0
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    channel1[channel1 < img_min] = img_min
    channel1[channel1 > img_max] = img_max
    channel1 = channel1 - img_min  # make pixel values start at 0

    # third channel (soft tissue)
    channel2 = shifted_img.copy()
    window_center = 40.0
    window_width = 380.0
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    channel2[channel2 < img_min] = img_min
    channel2[channel2 > img_max] = img_max
    channel2 = channel2 - img_min  # make pixel values start at 0

    # crop and resize
    full_size_img = np.zeros((*shifted_img.shape, 3))
    full_size_img[:, :, 0] = channel0
    full_size_img[:, :, 1] = channel1
    full_size_img[:, :, 2] = channel2
    cropped_img = crop_and_square(full_size_img)
    resized_img = cv2.resize(cropped_img, resize_dim)
    out_img[:, :, 0] = resized_img[:, :, 0]
    out_img[:, :, 1] = resized_img[:, :, 1]
    out_img[:, :, 2] = resized_img[:, :, 2]

    return out_img, pixel_stats


def preprocess_and_save(filenames, load_dir, save_dir,
                        data_df, img_stats_file,
                        resize_dim=(RESIZE, RESIZE)):
    """
    Preprocess all images with apply_slope_intercept_rescale_make_channels,
    and save the results in PNG format to specified directory.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(DATA_BASE_PATH + PNG_DIR + img_stats_file):
        with open(DATA_BASE_PATH + PNG_DIR + img_stats_file, 'w') as pxst:
            pxst.write("filename,patient_ID,study_instance_ID,series_instance_ID,study_ID,bits_allocated,bits_stored,pixel_representation,window_center,window_width,intercept,slope," + ",".join("pixel_perc_"+str(perc) for perc in np.linspace(1, 100, 19)) + "\n")

    for filename in tqdm(filenames):
        path = load_dir + filename
        png_name = filename.replace('.dcm', '.png')
        new_path = save_dir + png_name
        if os.path.exists(new_path):
            # if png image already exists do nothing
            continue
        try:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array
            dicom_fields = get_dicom_fields(dcm)
            # rescale, crop background, pad to square, make 3 channels
            img, pixel_stats = apply_slope_intercept_rescale_make_channels(
                img, dicom_fields, resize_dim)
            # save preprocessed image
            cv2.imwrite(new_path, img)
            # save its pixel statistics
            with open(DATA_BASE_PATH + PNG_DIR + img_stats_file, 'a') as pxst:
                pxst.write(png_name + "," + ",".join(str(v) for v in dicom_fields.values())
                           + ","  + ",".join(str(px) for px in pixel_stats) + '\n')
        except Exception as e:
            print("Failed for {} with message:\n{}".format(filename, e))
            data_df = data_df[data_df.filename != png_name]

    return data_df


if __name__ == "__main__":
    # Load CSVs
    train_df = pd.read_csv(DATA_BASE_PATH + 'stage_2_train.csv')
    sub_df = pd.read_csv(DATA_BASE_PATH + 'stage_2_sample_submission.csv')
    # Prepare dataframes
    train_df['filename'] = train_df['ID'].apply(
        lambda st: "ID_" + st.split('_')[1] + ".png")
    train_df['type'] = train_df['ID'].apply(
        lambda st: st.split('_')[2])
    sub_df['filename'] = sub_df['ID'].apply(
        lambda st: "ID_" + st.split('_')[1] + ".png")
    sub_df['type'] = sub_df['ID'].apply(
        lambda st: st.split('_')[2])
    test_df = pd.DataFrame(sub_df.filename.unique(),
                           columns=['filename'])
    pivot_df = train_df[['Label', 'filename', 'type']].drop_duplicates().pivot(
        index='filename', columns='type', values='Label').reset_index()

    # Preprocess all images and save as png:
    pivot_df = preprocess_and_save(filenames=os.listdir(DATA_BASE_PATH + TRAIN_DIR),
                                   load_dir=DATA_BASE_PATH + TRAIN_DIR,
                                   save_dir=DATA_BASE_PATH + PNG_DIR + "train/",
                                   data_df=pivot_df, img_stats_file=TRAIN_IMG_STATS_FILE,
                                   resize_dim=(RESIZE, RESIZE))
    test_df = preprocess_and_save(filenames=os.listdir(DATA_BASE_PATH + TEST_DIR),
                                  load_dir=DATA_BASE_PATH + TEST_DIR,
                                  save_dir=DATA_BASE_PATH + PNG_DIR + "test/",
                                  data_df=test_df, img_stats_file=TEST_IMG_STATS_FILE,
                                  resize_dim=(RESIZE, RESIZE))
    # Store the dataframes too:
    pivot_df.to_csv(DATA_BASE_PATH + PNG_DIR + "pivot_df.csv")
    test_df.to_csv(DATA_BASE_PATH + PNG_DIR + "test_df.csv")

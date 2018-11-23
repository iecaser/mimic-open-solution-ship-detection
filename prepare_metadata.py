import os
import pandas as pd
import numpy as np
from common_blocks import utils
from tqdm import tqdm
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import ipdb
import cv2

PARAM = utils.read_yaml('config.yaml').parameters
LOGGER = utils.get_logger('preparing-data')
if not os.path.exists(PARAM.masks_dir):
    LOGGER.info('made mask_dir: {}'.format(PARAM.masks_dir))
    os.mkdir(PARAM.masks_dir)
ORIGINAL_SIZE = (768, 768)


def prepare_mask():
    LOGGER.info('preparing mask images...')
    overlay_masks(annotation_file=PARAM.annotation_file,
                  masks_dir=PARAM.masks_dir,)


def prepare_metadata():
    metadata = generate_metadata(annotation_file=PARAM.annotation_file,
                                 train_images_dir=PARAM.train_images_dir,
                                 masks_dir=PARAM.masks_dir,
                                 test_images_dir=PARAM.test_images_dir,)
    metadata.to_csv(PARAM.metadata_filepath, sep=',', index=False)
    LOGGER.info('saved metadata.')


def overlay_masks(annotation_file, masks_dir):
    annotations = pd.read_csv(annotation_file, sep=',')
    for image_id, mask_codes in tqdm(annotations.groupby('ImageId')):
        if len(mask_codes.dropna()) == 0:
            continue
        mask = decode_masks(mask_codes.EncodedPixels)
        joblib.dump(value=mask, filename=os.path.join(masks_dir, image_id.split('.')[0]))


def decode_masks(mask_codes):
    mask = np.zeros(ORIGINAL_SIZE[0] * ORIGINAL_SIZE[1], dtype=np.uint8)
    for mask_code in mask_codes:
        mask_code = np.array(mask_code.split()).astype('int32')
        starts, lengths = mask_code[::2], mask_code[1:][::2]
        ends = starts + lengths
        for start, end in zip(starts, ends):
            mask[start:end] += 1
    mask = mask.reshape(ORIGINAL_SIZE[1], ORIGINAL_SIZE[0]).T
    return mask


def generate_metadata(annotation_file, train_images_dir, masks_dir, test_images_dir):
    metadata = {}
    annotations = pd.read_csv(annotation_file, sep=',')
    LOGGER.info('preparing metadata(train)...')
    for filename in tqdm(os.listdir(train_images_dir)):
        image_id = filename.split('.')[0]
        number_of_ships = len(annotations.query('ImageId==@filename').dropna())
        metadata.setdefault('id', []).append(image_id)
        metadata.setdefault('file_path_image', []).append(os.path.join(train_images_dir, filename))
        metadata.setdefault('is_train', []).append(1)
        metadata.setdefault('file_path_mask', []).append(os.path.join(masks_dir, image_id))
        metadata.setdefault('number_of_ships', []).append(number_of_ships)
        metadata.setdefault('is_not_empty', []).append(int(number_of_ships != 0))

    LOGGER.info('preparing metadata(test)...')
    for filename in tqdm(os.listdir(test_images_dir)):
        image_id = filename.split('.')[0]
        metadata.setdefault('id', []).append(image_id)
        metadata.setdefault('file_path_image', []).append(os.path.join(test_images_dir, filename))
        metadata.setdefault('is_train', []).append(0)
        metadata.setdefault('file_path_mask', []).append(None)
        metadata.setdefault('number_of_ships', []).append(None)
        metadata.setdefault('is_not_empty', []).append(None)
    metadata = pd.DataFrame(metadata)
    return metadata


prepare_mask()
prepare_metadata()

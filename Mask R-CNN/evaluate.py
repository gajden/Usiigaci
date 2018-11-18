"""


Usage:
    evaluate.py --models-dir=<models_dir> --test-dir=<test_dir>
"""
import os
import os.path as op
import sys
from collections import defaultdict

import numpy as np
import cv2
from docopt import docopt
from tqdm import tqdm
from train import cellConfig
from train import cellDataset
from mrcnn import utils
from mrcnn import model as modellib


class CellInferenceConfig(cellConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


def create_dataset(data_dir):
    """

    :param data_dir:
    :return:
    """
    val_dataset = cellDataset()
    val_dataset.load_cell(data_dir)
    val_dataset.prepare()

    return val_dataset


def find_all_models(data_dir):
    """
    Searches through the whole directory structure
    for *.h5 or *.hdf5 files. Returns paths to this files.
    :param data_dir: string - path to directory.
    :return:
    """
    try:
        assert op.exists(data_dir) and op.isdir(data_dir)
    except AssertionError:
        print("Selected models dir does not exist or is invalid.")
        sys.exit(0)

    model_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.h5') or file.endswith('.hdf5'):
                model_paths.append(op.join(root, file))
    return model_paths


def evaluate_models(models_paths, dataset):
    """

    :return:
    """
    models_scores = defaultdict(list)

    config = CellInferenceConfig()

    model = modellib.MaskRCNN(mode="inference", config=config, model_dir='./')

    best_map = 0
    best_model = None

    for model_path in models_paths:
        print('Model:', model_path)
        model.load_weights(model_path, by_name=True)

        for image_id in tqdm(dataset.image_ids):
            image = dataset.load_image(image_id)
            gt_mask, gt_class_id = dataset.load_mask(image_id)
            gt_bbox = utils.extract_bboxes(gt_mask)

            result = model.detect([image], verbose=0)[0]

            AP, precisions, recalls, overlaps = \
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                 result['rois'], result['class_ids'],
                                 result['scores'], result['masks'])
            models_scores[model_path].append(AP)
        map = np.mean(models_scores[model_path])
        print("> mAP @ IoU=0.5: ", map)
        if map > best_map:
            best_map = map
            best_model = model_path

    print('\nBest model: %s' % best_model)
    print('\t> with mAP @ IoU=0.5: ', best_map)


def main():
    args = docopt(__doc__)

    models_dir = args['--models-dir']
    test_dir = args['--test-dir']

    dataset = create_dataset(test_dir)
    models_path = find_all_models(models_dir)

    evaluate_models(models_path, dataset)


if __name__ == '__main__':
    main()

"""
Training script for custom datasets. Only required argument is config file.
Other can be used to overwrite some options from config. Final config used
for training will be saved in training directory with trained weights.

Usage:
    train_custom_dataset.py config_file [--weights-file=<weights>] [--save-dir=<save_dir>]
                                        [--iter-nums=<iter>] [--save-freq=<freq>] [--lr=<lr>]
                                        [--dataset-dir=<dataset_dir>]

"""

# TODO save only best model option
# custom dataset name


import os
import os.path as op
import random

import torch
from torch import nn
from PIL import Image
import json
import logging
import torch
import numpy as np
import skimage.draw as draw
import tempfile
from pycocotools.coco import COCO
import cv2
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

from maskrcnn_benchmark.data.build import *
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

import torch.distributed as dist

from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torchvision import transforms as T
from torchvision.transforms import functional as F
# from google.colab.patches import cv2_imshow


import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config


try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


class CellDataset(object):
    def __init__(self, data_dir, transforms=None, inputs_name='phase.png',
                 annotations_name='labeled.png', use_cache=True, cache_path='cells_coco.json',
                 use_random_crop=False, crop_size=(512, 512)):
        self.use_cache = use_cache
        self.cache_path = cache_path

        self.data_dir = data_dir
        self.inputs_name = inputs_name
        self.annotations_name = annotations_name

        self.use_random_crop = use_random_crop
        self.crop_size = crop_size

        self.transforms = transforms

        self.image_info = []
        self.logger = logging.getLogger(__name__)

        self.class_names = {'cell': 1}

        self.ids = sorted([input_id for input_id in os.listdir(self.data_dir) if
                           op.isdir(op.join(self.data_dir, input_id))])
        self.num_examples = len(self.ids)

        for img_id in self.ids:
            img_path = op.join(self.data_dir, img_id, self.inputs_name)

            # print(img_path)
            #             print(type(img))
            img = cv2.imread(img_path)

            self.image_info.append({
                'img_path': img_path,
                'anno_path': op.join(self.data_dir, img_id, self.annotations_name),
                'width': img.shape[1], 'height': img.shape[0]
            })

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.contiguous_category_id_to_json_id = {0: 0, 1: 1}

        self.prepare_for_evaluation()

    def __len__(self):
        return self.num_examples

    def __getitem__(self, index):
        img, target = self.get_groundtruth(index)
        target.clip_to_image(remove_empty=False)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def get_groundtruth(self, index):
        img_id = self.id_to_img_map[index]

        img_path = op.join(self.data_dir, img_id, self.inputs_name)

        cv_img = cv2.imread(img_path)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv_img)

        if self.use_random_crop:
            width, height = img.size
            x_start = random.randint(0, width - self.crop_size[0] - 1)
            y_start = random.randint(0, height - self.crop_size[1] - 1)

            img = img.crop((x_start, y_start, x_start + self.crop_size[0], y_start + self.crop_size[1]))

        img_info = self.image_info[index]

        # Extract binary masks
        if self.use_random_crop:
            color_mask = cv2.imread(img_info['anno_path'])[y_start: y_start + self.crop_size[1],
                                                           x_start: x_start + self.crop_size[0]]
        else:
            color_mask = cv2.imread(img_info['anno_path'])

        # Extract bboxes
        instances, bboxes, classes_ids = self.__preprocess_annotation(color_mask)
        bboxes = torch.as_tensor(bboxes).reshape(-1, 4)
        target = BoxList(bboxes, (img_info['width'], img_info['height']), mode='xyxy')

        target.add_field('labels', classes_ids)
        masks = SegmentationMask(instances, (img_info['width'], img_info['height']), mode='mask')
        target.add_field('masks', masks)

        return img, target

    def __preprocess_annotation(self, color_mask, ):
        instances_masks = []
        instances_bboxes = []
        classes_ids = []

        background_mask = None

        if background_mask is None:
            background_mask = np.ones(color_mask.shape[:2])

        for dim in range(color_mask.shape[2]):
            background_mask[color_mask[:, :, dim] != 0] = 0

        colors = np.unique(color_mask.reshape(-1, color_mask.shape[-1]),
                           axis=0, return_counts=True)[0]

        tmp_mask = np.zeros(background_mask.shape)
        tmp_mask[np.where(background_mask == 1)] = 255

        for color in colors:
            if not np.all(color == [0, 0, 0]):
                binary_mask = np.zeros(color_mask.shape[:2])
                binary_mask[np.where(np.all(color_mask == color, axis=-1))] = 1

                instance_loc = np.where(binary_mask == 1)
                x_min, x_max = min(instance_loc[1]), max(instance_loc[1])
                y_min, y_max = min(instance_loc[0]), max(instance_loc[0])
                instances_bboxes.append([x_min, y_min, x_max, y_max])

                instances_masks.append(binary_mask)
                classes_ids.append(1)

        #         instances_masks.append(background_mask)
        #         classes_ids.append(0)

        classes_ids = torch.from_numpy(np.array(classes_ids))
        return torch.from_numpy(np.asarray(instances_masks, dtype=np.uint8)), instances_bboxes, classes_ids

    def get_img_info(self, index):
        return self.image_info[index]

    def prepare_for_evaluation(self):
        print('Prepare for evaluation')
        images = []
        annotations = []
        results = []

        use_random_crop = self.use_random_crop
        self.use_random_crop = False

        if self.use_cache and op.exists(self.cache_path):
            self.coco = COCO(self.cache_path)

        else:
            categories = [{"id": 1, "name": "cell"}]

            i = 1
            ann_id = 0

            for img_id, d in tqdm(enumerate(self.ids)):
                info = self.image_info[img_id]
                images.append({"id": img_id, 'height': info['height'], 'width': info['width']})

                _, target = self.get_groundtruth(img_id)

                for bbox in target.bbox:
                    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    bbox = [float(bbox[0]), float(bbox[1]),
                            float(width), float(height)]  # x, y, width, height
                    area = width * height

                    annotations.append({
                        'id': int(ann_id),
                        'category_id': 1,
                        'image_id': int(img_id),
                        'area': float(area),
                        'bbox': bbox,
                        'iscrowd': 0
                    })
            with open(self.cache_path, 'w') as f_out:
                json.dump({
                    "images": images,
                    "annotations": annotations,
                    "categories": categories
                }, f_out, indent=4)
            self.coco = COCO(self.cache_path)
        self.use_random_crop = use_random_crop


def build_data_loader(cfg, dataset, is_train=True, is_distributed=False, start_iter=0):
    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (
                images_per_batch % num_gpus == 0
        ), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_batch = cfg.TEST.IMS_PER_BATCH
        assert (
                images_per_batch % num_gpus == 0
        ), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(
            images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0

    if images_per_gpu > 1:
        logger = logging.getLogger(__name__)
        logger.warning(
            "When using more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH (for training) or "
            "TEST.IMS_PER_BATCH (for inference). For training, you must "
            "also adjust the learning rate and schedule length according "
            "to the linear scaling rule. See for example: "
            "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
        )

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []

    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST

    # If bbox aug is enabled in testing, simply set transforms to None and we will apply transforms later
    transforms = None if not is_train and cfg.TEST.BBOX_AUG.ENABLED else build_transforms(cfg, is_train)

    dataset.transforms = transforms
    datasets = [dataset]

    data_loaders = []
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter
        )
        collator = BBoxAugCollator() if not is_train and cfg.TEST.BBOX_AUG.ENABLED else \
            BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)
    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders


def train(cfg, local_rank, distributed, dataset):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
#     extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
#     arguments.update(extra_checkpoint_data)

    data_loader = build_data_loader(
        cfg,
        dataset
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
    )

    return model


def main():
    num_gpus = 1
    distributed = num_gpus > 1

    config_file = 'cell_config.yaml'
    local_rank = 1

    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(config_file)
    #     cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(config_file))
    with open(config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    dataset = CellDataset('/home/bengio/cells/Usiigaci/Mask R-CNN/train', inputs_name='raw.tif', use_random_crop=True)
    model = train(cfg, local_rank, distributed, dataset)

    # if not args.skip_test:
    #     run_test(cfg, model, args.distributed)


if __name__ == '__main__':
    main()


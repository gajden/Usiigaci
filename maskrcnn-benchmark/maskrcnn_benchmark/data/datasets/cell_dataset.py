import json
import logging
import os
import os.path as op

import numpy as np
import cv2
import torch
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


class CellDataset(object):
    def __init__(self, data_dir, transforms=None, inputs_name='phase.png',
                 annotations_name='instances_ids.png', use_cache=True, cache_path='cells_coco.json',
                 use_random_crops=False, crop_size=(512, 512)):
        self.use_cache = use_cache
        self.cache_path = cache_path

        self.data_dir = data_dir
        self.inputs_name = inputs_name
        self.annotations_name = annotations_name

        self.use_random_crops = use_random_crops
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
            # TODO fix this, so there's no explicit cropping
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
        img_id = self.id_to_img_map[index]

        img_path = op.join(self.data_dir, img_id, self.inputs_name)

        cv_img = cv2.imread(img_path)[:1022, :1022]
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv_img)
        #         img = Image.open(img_path).convert('RGB')

        target = self.get_groundtruth(index)
        target.clip_to_image(remove_empty=False)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def get_groundtruth(self, index):
        img_info = self.image_info[index]

        # Extract binary masks
        color_mask = cv2.imread(img_info['anno_path'])

        # Extract bboxes
        instances, bboxes, classes_ids = self.__preprocess_annotation(color_mask)
        bboxes = torch.as_tensor(bboxes).reshape(-1, 4)
        target = BoxList(bboxes, (img_info['width'], img_info['height']), mode='xyxy')

        target.add_field('labels', classes_ids)
        masks = SegmentationMask(instances, (img_info['width'], img_info['height']), mode='mask')
        target.add_field('masks', masks)

        return target

    def __preprocess_annotation(self, color_mask):
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

        if self.use_cache and op.exists(self.cache_path):
            self.coco = COCO(self.cache_path)

        else:
            categories = [{"id": 1, "name": "cell"}]

            i = 1
            ann_id = 0

            for img_id, d in tqdm(enumerate(self.ids)):
                info = self.image_info[img_id]
                images.append({"id": img_id, 'height': info['height'], 'width': info['width']})

                target = self.get_groundtruth(img_id)

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


class MultiClassCellDataset(object):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.images_dir = op.join(data_dir, 'images')
        self.annotations_dir = op.join(data_dir, 'annotations')

        self.transforms = transforms

        # create categories dict

        self.ids = []  # images ids (names, paths, etc.)
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        self.img_info = None
        self.__init_dataset()

    def __init_dataset(self):
        class_names = set()

        images_names = os.listdir(self.images_dir)
        for idx, image_name in enumerate(images_names):
            image_annotation_dir = op.join(self.annotations_dir, op.splitext(image_name)[0])

            annotations_names = os.listdir(image_annotation_dir)
            class_names = class_names.union(set([op.splitext(name)[0] for name in annotations_names]))
            annotations_paths = [op.join(image_annotation_dir, name)
                                 for name in annotations_names]
            self.add_image('custom', idx, op.join(self.images_dir, image_name), annotations=annotations_paths)

        self.classes = tuple(sorted(class_names))

    def __getitem__(self, idx):
        # load the image as a PIL Image
        image = ...

        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        boxes = [[0, 0, 10, 10], [10, 20, 50, 50]]
        # and labels
        labels = torch.tensor([10, 20])

        # create a BoxList from the boxes
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)

        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)

        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx

    def get_img_info(self, idx):
        return self.img_info[idx]


def test():
    data_dir = ''
    dataset = CellDataset(data_dir)


class MyDataset(object):
    def __init__(self):
        # as you would do normally
        pass

    def __getitem__(self, idx):
        # load the image as a PIL Image
        image = ...

        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        boxes = [[0, 0, 10, 10], [10, 20, 50, 50]]
        # and labels
        labels = torch.tensor([10, 20])

        # create a BoxList from the boxes
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)

        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)

        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        return {"height": img_height, "width": img_width}


if __name__ == '__main__':
    test()

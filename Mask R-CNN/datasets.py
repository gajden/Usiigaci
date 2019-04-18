import os
import os.path as op
import random
from math import sqrt, ceil, floor

from tqdm import tqdm
import numpy as np
import cv2

from mrcnn.utils import Dataset
from mrcnn.visualize import random_colors


class MultiClassCellDataset(Dataset):
    """
    Dataset for handling cells with multiple classes and instances for each class.
    Dataset structure:
        <dataset_dir>
            images
                <sample1_name>.[png, jpg]
                <sample2_name>.[png, jpg]
                ...
            annotations
                <sample1_name>
                    <class1>.png
                    <class2>.png
                    ...
                <sample2_name>
                    <class1>.png
                    <class2>.png
    """
    def __init__(self):
        super(MultiClassCellDataset, self).__init__()

        self.dataset_dir = None

        self.images_dir = None
        self.annotations_dir = None

    def init_dataset(self, dataset_dir):
        """

        :param dataset_dir: string - path to dataset directory
        :return:None
        """
        self.dataset_dir = dataset_dir

        class_names = set()
        self.images_dir = op.join(dataset_dir, 'images')
        self.annotations_dir = op.join(dataset_dir, 'annotations')

        # Look up for all class names
        images_names = os.listdir(self.images_dir)
        for idx, image_name in enumerate(images_names):
            image_annotation_dir = op.join(self.annotations_dir, op.splitext(image_name)[0])

            annotations_names = os.listdir(image_annotation_dir)
            class_names = class_names.union(set([op.splitext(name)[0] for name in annotations_names]))
            annotations_paths = [op.join(image_annotation_dir, name)
                                 for name in annotations_names]
            self.add_image('custom', idx, op.join(self.images_dir, image_name), annotations=annotations_paths)

        for i, class_name in enumerate(sorted(class_names)):
            self.add_class(class_name, i + 1, class_name)
        self.prepare()

    def prepare(self):
        """Prepares the Dataset class for use.

            TODO: class map is not supported yet. When done, it should handle mapping
                  classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

        self.class_from_source_map = {key.split('.')[0]: val for key, val in self.class_from_source_map.items()}

    def load_mask(self, image_id):
        """

        :param image_id:
        :return:
        """
        image_info = self.image_info[image_id]

        instances_masks = []
        instances_classes = []

        for annotation_file in image_info['annotations']:
            class_name = op.splitext(op.basename(annotation_file))[0]
            class_id = self.class_from_source_map[class_name]

            color_mask = cv2.imread(annotation_file)

            colors = np.unique(color_mask.reshape(-1, color_mask.shape[-1]),
                               axis=0, return_counts=True)[0]

            for color in colors:
                binary_mask = np.zeros(color_mask.shape[:2])

                binary_mask[np.where(np.all(color_mask == color, axis=-1))] = 1

                instances_masks.append(binary_mask)
                instances_classes.append(class_id)

        instances_masks = np.swapaxes(np.swapaxes(np.asarray(instances_masks), 0, 2), 0, 1)
        return instances_masks, np.asarray(instances_classes)


def get_n_colors(n):
    ret = []
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        ret.append((r, g, b))
    return ret


def generate_stub_data(dataset_dir, samples_num, instances_num=300, classes=['cell', 'boundary'], mask_size=500):
    """
    Generates stub dataset for testing

    Data input format:
    <dataset_dir>
        <images>
            sample1_name.[png, jpg]
            sample2_name.[png, jpg]
            ...
        <annotations>
            <sample1_name>
                class1.png
                class2.png
                ...
            <sample2_name>
                class1.png
                class2.png
                ...
            <sample3_dir>
                ...
    :param dataset_dir: string - path to dataset directory, will be created
        if doesn't exist
    :param samples_num: int - number of samples to generate
    :param instances_num:
    :param classes: list of strings, each string is the name of class
    :param mask_size:
    :return:
    """
    alpha = 0.5
    os.makedirs(dataset_dir, exist_ok=True)
    colors = random_colors(instances_num)

    images_dir = op.join(dataset_dir, 'images')
    annotations_dir = op.join(dataset_dir, 'annotations')

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    for i in tqdm(range(samples_num)):
        image = np.ones((mask_size, mask_size, 3))
        cv2.imwrite(op.join(images_dir, 'sample_%d.jpg' % i), image)

        sample_dir = op.join(annotations_dir, 'sample_%d' % i)
        os.makedirs(sample_dir, exist_ok=True)

        # generate grid
        grid_size = ceil(sqrt(instances_num))

        cell_size = floor(mask_size / grid_size)

        classes_masks = [np.zeros((mask_size, mask_size, 3)) for _ in classes]
        # print(colors)
        for j in range(instances_num):
            row = j // grid_size
            column = j % grid_size

            color = [int((1 - alpha) + alpha * c * 255) for c in colors[j]]
            # print(j, color)

            instance_x = cell_size * column
            instance_y = cell_size * row

            class_width = int(cell_size / len(classes))
            for k, inst_class in enumerate(classes):
                classes_masks[k][instance_y: instance_y + cell_size, instance_x + k * class_width: instance_x + (k + 1) * class_width] = color

        for c, class_name in enumerate(classes):
            cv2.imwrite(op.join(sample_dir, '%s.png' % class_name), classes_masks[c])

# imports
import json
import time
import pickle
import scipy.misc
import skimage.io
import caffe

import numpy as np
import os.path as osp

from xml.dom import minidom
from random import shuffle
from threading import Thread
from PIL import Image

from tools import SimpleTransformer


class InstanceSegmentationDataLayerSync(caffe.Layer):
    """
    This is a simple syncronous datalayer for training a instance segmentation model on
    PASCAL.
    """

    def setup(self, bottom, top):
        self.top_names = ['data', 'label']

        # === Read input parameters ===

        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)

        # Check the paramameters for validity.
        check_params(params)

        # store input as class variables
        self.batch_size = params['batch_size']

        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader(params, None)

        # === reshape tops ===
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(
            self.batch_size, 3, params['im_shape'][0], params['im_shape'][1])
        # Note the 20 channels (because PASCAL has 20 classes.)
        # top[1].reshape(self.batch_size, 20)
        top[1].reshape(
            self.batch_size, 1, params['seg_shape'][0], params['seg_shape'][1])

        print_info("InstanceSegmentationDataLayerSync", params)

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, seg = self.batch_loader.load_next_image()

            # Add directly to the caffe data layer
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = seg

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


class BatchLoader(object):
    """
    This class abstracts away the loading of images.
    Images can either be loaded singly, or in a batch. The latter is used for
    the asyncronous data layer to preload batches while other processing is
    performed.
    """

    def __init__(self, params, result):
        self.result = result
        self.batch_size = params['batch_size']
        self.pascal_root = params['pascal_root']
        self.im_shape = params['im_shape']
        self.seg_shape = params['seg_shape']
        # get list of image indexes.
        list_file = params['split'] + '.txt'
        self.indexlist = [line.rstrip('\n') for line in open(
            osp.join(self.pascal_root, 'ImageSets/Segmentation', list_file))]
        self._cur = 0  # current image
        # this class does some simple data-manipulations
        self.transformer = SimpleTransformer()

        print "BatchLoader initialized with {} images".format(
            len(self.indexlist))

    def load_next_image(self):
        """
        Load the next image in a batch.
        """
        # Did we finish an epoch?
        if self._cur == len(self.indexlist):
            self._cur = 0
            shuffle(self.indexlist)

        # Load an image
        index = self.indexlist[self._cur]  # Get the image index
        image_file_name = index + '.jpg'
        im = np.asarray(Image.open(
            osp.join(self.pascal_root, 'JPEGImages', image_file_name)))
        im = scipy.misc.imresize(im, self.im_shape)  # resize

        # Load and prepare ground truth segmentation
        segment_file_name = index + '.png'
        seg = np.asarray(Image.open(
            osp.join(self.pascal_root, 'SegmentationObject', segment_file_name)))
        seg = scipy.misc.imresize(seg, self.im_shape, 'nearest')

        # do a simple horizontal flip as data augmentation
        flip = np.random.choice(2) * 2 - 1
        im = im[:, ::flip, :]
        seg = seg[:, ::flip]

        # # Load and prepare ground truth
        # start = time.time()
        # label_map_org = np.zeros((self.im_shape[0], self.im_shape[1])).astype(np.float32)
        # label_map_org[seg == 255] = 255
        # maxlabel = np.max(seg[seg < 255])
        # centroids = np.zeros((maxlabel, 2)).astype(np.float32)
        # npts = np.zeros((maxlabel, 1)).astype(np.float32)
        # for y in range(self.im_shape[0]):
        #     for x in range(self.im_shape[1]):
        #         label = seg[y, x]
        #         if 0 < label < 255:
        #             centroids[label - 1, :] += [y, x]
        #             npts[label - 1] += 1

        # for label in range(maxlabel):
        #     # assign an id based on centroid
        #     if npts[label] > 0:
        #         centroids[label, :] /= npts[label] * 32
        #         label_map_org[seg == label + 1] = int(round(centroids[label, 0])) * 8 + int(round(centroids[label, 1])) + 1
        # label_map = scipy.misc.imresize(label_map_org, self.seg_shape, 'nearest')
        # end = time.time()
        # print end - start

        # Load and prepare ground truth
        # start = time.time()
        label_map_org = np.zeros((self.im_shape[0], self.im_shape[1])).astype(np.int)
        label_map_org[seg == 255] = 255
        maxlabel = np.max(seg[seg < 255])
        unique_labels = np.unique(seg[seg < 255])
        centroids = np.zeros((maxlabel+1, 2)).astype(np.float32)
        x_coords = np.tile(np.arange(self.im_shape[1]), (self.im_shape[0], 1))
        y_coords = np.tile(np.arange(self.im_shape[0]).reshape((self.im_shape[0], 1)), (1, self.im_shape[1]))
        # for label in range(maxlabel):
        for label in unique_labels:
            if label == 0:
                continue
            mask = np.zeros((self.im_shape[0], self.im_shape[1])).astype(np.float32)
            # mask[seg == label + 1] = 1
            mask[seg == label] = 1
            center_x = np.mean(x_coords[mask == 1])
            center_y = np.mean(y_coords[mask == 1])
            centroids[label, :] = [center_y, center_x]
        # for label in range(maxlabel):
        for label in unique_labels:
            if label == 0:
                continue
            centroids[label, :] /= 32
            label_map_org[seg == label] = int(round(centroids[label, 0])) * 8 + int(
                round(centroids[label, 1])) + 1
        label_map = scipy.misc.imresize(label_map_org, self.seg_shape, 'nearest')
        # end = time.time()
        # print end - start
        # print label_map.shape

        self._cur += 1
        return self.transformer.preprocess(im), label_map


def load_pascal_annotation(index, pascal_root):
    """
    This code is borrowed from Ross Girshick's FAST-RCNN code
    (https://github.com/rbgirshick/fast-rcnn).
    It parses the PASCAL .xml metadata files.
    See publication for further details: (http://arxiv.org/abs/1504.08083).

    Thanks Ross!

    """
    classes = ('__background__',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
    class_to_ind = dict(zip(classes, xrange(21)))

    filename = osp.join(pascal_root, 'Annotations', index + '.xml')

    # print 'Loading: {}'.format(filename)

    def get_data_from_tag(node, tag):
        return node.getElementsByTagName(tag)[0].childNodes[0].data

    with open(filename) as f:
        data = minidom.parseString(f.read())

    objs = data.getElementsByTagName('object')
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, 21), dtype=np.float32)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        # Make pixel indexes 0-based
        x1 = float(get_data_from_tag(obj, 'xmin')) - 1
        y1 = float(get_data_from_tag(obj, 'ymin')) - 1
        x2 = float(get_data_from_tag(obj, 'xmax')) - 1
        y2 = float(get_data_from_tag(obj, 'ymax')) - 1
        cls = class_to_ind[
            str(get_data_from_tag(obj, "name")).lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = cls
        overlaps[ix, cls] = 1.0

    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_overlaps': overlaps,
            'flipped': False,
            'index': index}


def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    assert 'split' in params.keys(
    ), 'Params must include split (train, val, or test).'

    required = ['batch_size', 'pascal_root', 'im_shape']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


def print_info(name, params):
    """
    Ouput some info regarding the class
    """
    print "{} initialized for split: {}, with bs: {}, im_shape: {}.".format(
        name,
        params['split'],
        params['batch_size'],
        params['im_shape'])


if __name__ == '__main__':
    # pascal_root = '/home/zhaoliang07/data/VOCdevkit/VOC2012'
    pascal_root = './demo_images/VOC2012'
    params = dict(batch_size=128, im_shape=[256, 256], seg_shape=[8, 8], split='train',
                  pascal_root=pascal_root)
    batch_loader = BatchLoader(params, None)
    im, seg = batch_loader.load_next_image()
    im, seg = batch_loader.load_next_image()
    im, seg = batch_loader.load_next_image()
    im, seg = batch_loader.load_next_image()
    params = dict(batch_size=128, im_shape=[256, 256], seg_shape=[8, 8], split='val',
                  pascal_root=pascal_root)
    batch_loader = BatchLoader(params, None)
    im, seg = batch_loader.load_next_image()
    im, seg = batch_loader.load_next_image()
    im, seg = batch_loader.load_next_image()

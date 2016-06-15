import sys
sys.path.append('/home/yi/code/tools/caffe/python')
import caffe
import surgery, score

import numpy as np
import os

import setproctitle
setproctitle.setproctitle(os.path.basename(os.getcwd()))

# weights = '../vgg16fc.caffemodel'
# weights = '../VGG_ILSVRC_16_layers.caffemodel'
weights = '../VGG_ILSVRC_16_layers_conv.caffemodel'

# init
caffe.set_mode_gpu()
caffe.set_device(1)

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
# val = np.loadtxt('../data/segvalid11.txt', dtype=str)
val = np.loadtxt('/media/yi/DATA/data-orig/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt', dtype=str)

for _ in range(25):
    solver.step(4000)
    score.seg_tests(solver, False, val, layer='score')

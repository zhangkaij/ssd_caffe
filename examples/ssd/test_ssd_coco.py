import os
import caffe
import numpy as np
import matplotlib.pyplot as plt

#%matplotlib inline
#os.chdir('/home/zhangkj/software/caffe')

caffe.set_device(2)
caffe.set_mode_gpu()

solver = caffe.get_solver('models/VGGNet/coco/SSD_300x300/solver.prototxt')
solver.net.copy_from('models/VGGNet/coco/SSD_300x300/vgg_coco.caffemodel')

solver.net.forward(end='conv1_1')
#solver.net.blobs['data'].data.shape
#for i in range(3):
#    for j in range(300):
#        for p in range(300):
#            print(solver.net.blobs['data'].data[0, i, j, p])      

pytorch_data = []
with open('input.log', 'r') as f:
    for line in f.readlines():
        pytorch_data.append(float(line))
        
solver.net.blobs['data'].data[...] = np.array(pytorch_data).reshape(1, 3, 300, 300)
solver.net.forward(start='conv1_1')

#for i in range(797292):
#    print(solver.net.blobs['mbox_conf'].data[0, i])
#!/usr/bin/env python 
from __future__ import division

import os
from collections import OrderedDict

batch_size = 100
training_steps = 1500
img_height = 66
img_width = 200
img_channels = 3
write_summary = True
use_category_normal = False # if ture, center/curve images are equally selected.

# change this to the directory that contains the source videos
save_dir = os.path.abspath('MiniModelNew')
save_dir2 = os.path.abspath('MiniModelNew2')
save_dir3 = os.path.abspath('MiniModelNew3')
save_dir4 = os.path.abspath('MiniModelNew4')

out_dir = os.path.abspath('output')

shuffle_training = True


if not os.path.isdir(out_dir):
    os.makedirs(out_dir)

epochs = OrderedDict()
epochs['train'] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,21,22,23,24,25,26,27,28,29,30,32,33,34,43] # epoch_ids for training data: e.g., "epochs['train'] = [1, 2, 3]" 
epochs['val'] = [19,20,35,36,37,38,39,40,41,42] # epoch_ids for validation data: e.g., "epochs['val'] = [4]" 

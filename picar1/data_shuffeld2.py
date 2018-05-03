
from __future__ import division

import random
import os
import sys
from collections import OrderedDict
import cv2
import params
import preprocess
import local_common as cm

################ parameters ###############


epochs = params.epochs
img_height = params.img_height
img_width = params.img_width
img_channels = params.img_channels

purposes = ['train', 'val']
imgs = OrderedDict()
wheels = OrderedDict()
for purpose in purposes:
    imgs[purpose] = []
    wheels[purpose] = []

categories = ['center', 'curve']    
imgs_cat = OrderedDict()
wheels_cat = OrderedDict()
for p in purposes:
    imgs_cat[p] = OrderedDict()
    wheels_cat[p] = OrderedDict()
    for c in categories:
        imgs_cat[p][c] = []
        wheels_cat[p][c] = []
    
# load all preprocessed training images into memory
def upload_files():
  from google.colab import files
  uploaded = files.upload()
  for k, v in uploaded.items():
    open(k, 'wb').write(v)
  return list(uploaded.keys())
def load_imgs():
    global imgs
    global wheels

    for p in purposes:
        for epoch_id in epochs[p]:
		    
            print ('processing and loading "{}" epoch {} into memory, current num of imgs is {}...'.format(
                p, epoch_id, len(imgs[p])))

            # vid_path = cm.jn(data_dir, 'epoch{:0>2}_front.mkv'.format(epoch_id))
            data_dir = os.path.abspath('training_images{}'.format(epoch_id))
            
            assert os.path.isdir(data_dir)

            print ("upload images from:", data_dir)
           
            

            # csv_path = cm.jn(data_dir, 'epoch{:0>2}_steering.csv'.format(epoch_id))
            csv_path = os.path.abspath('out-key{}.csv'.format(epoch_id))
            assert os.path.isfile(csv_path)

            print ("upload ground truth from:", csv_path)
            rows = cm.fetch_csv_data(csv_path)
            print (len(rows), frame_count)
            assert frame_count == len(rows)
            yy = [[float(row['wheel'])] for row in rows]

            for image in os.listdir(data_dir):
                img = cv2.imread(os.path.join(data_dir,image))
                img = preprocess.preprocess(img)
                imgs[p].append(img)

            wheels[p].extend(yy)
            assert len(imgs[p]) == len(wheels[p])

            

def load_batch(purpose):
    p = purpose
    assert len(imgs[p]) == len(wheels[p])
    n = len(imgs[p])
    assert n > 0

    ii = random.sample(xrange(0, n), params.batch_size)
    assert len(ii) == params.batch_size

    xx, yy = [], []
    for i in ii:
        xx.append(imgs[p][i])
        yy.append(wheels[p][i])
    xx_img = np.asarray(xx)
	yy_img = np.asarray(yy)
    return xx_img, yy_img

def categorize_imgs():
    global imgs
    global wheels
    global imgs_cat
    global wheels_cat

    for p in purposes:
        n = len(imgs[p])

        for i in range(n):
            # print 'wheels[{}][{}]:{}'.format(p, i, wheels[p][i])
            if abs(wheels[p][i][0]) != 30:
                imgs_cat[p]['center'].append(imgs[p][i])
                wheels_cat[p]['center'].append(wheels[p][i])
            else:
                imgs_cat[p]['curve'].append(imgs[p][i])
                wheels_cat[p]['curve'].append(wheels[p][i])

        print ('---< {} >---'.format(p))
        for c in categories:
            print ('# {} imgs: {}'.format(c, len(imgs_cat[p][c])))

def load_batch_category_normal(purpose):
    p = purpose
    xx, yy = [], []
    nc = len(categories)
    for c in categories:
        n = len(imgs_cat[p][c])
        assert n > 0
        ii = random.sample(xrange(0, n), int(params.batch_size/nc))
        assert len(ii) == int(params.batch_size/nc)
        for i in ii:
            xx.append(imgs_cat[p][c][i])
            yy.append(wheels_cat[p][c][i])

    return xx, yy

if __name__ == '__main__':
    load_imgs()

    load_batch()
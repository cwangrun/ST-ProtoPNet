import os
import pandas as pd
from PIL import Image
from shutil import copyfile
import numpy as np
import scipy.io

def makedir(path):
    if not os.path.exists(path): 
        os.makedirs(path)

# set paths
rootpath = '/home/Downloads/'
imgspath = rootpath + 'car_ims/'
trainpath = '/home/Downloads/Cars_cropped/train_cropped/'
testpath = '/home/Downloads/Cars_cropped/test_cropped/'

# read img names, bounding_boxes
label_names = '/home/Downloads/cars_annos.mat'
mat = scipy.io.loadmat(label_names)['annotations'][0]
train_test_indicator = np.array([int(info[-1]) for info in mat])   # 0 training, 1 test
boxs = np.array([[int(info[1]), int(info[2]), int(info[3]), int(info[4])] for info in mat])
classes = np.array([int(info[-2]) for info in mat])
names = np.array([str(info[0][0]) for info in mat])


# crop imgs
for i in range(len(train_test_indicator)):
    im = Image.open(os.path.join(rootpath, names[i]))

    box = boxs[i]

    # im_crop_ = im.crop((10, 100, 60, 120))

    im_crop = im.crop((box[0], box[1], box[2], box[3]))

    if classes[i] < 10:
        class_name = '00' + str(classes[i])
    elif classes[i] < 100:
        class_name = '0' + str(classes[i])
    else:
        class_name = str(classes[i])

    if train_test_indicator[i] == 0:
        save_path = os.path.join(trainpath, class_name, names[i].split('/')[-1])
    else:
        save_path = os.path.join(testpath, class_name, names[i].split('/')[-1])

    makedir(os.path.dirname(save_path))

    im_crop.save(save_path, quality=95)

    print('{} imgs cropped and saved.'.format(i + 1))

print('All Done.')



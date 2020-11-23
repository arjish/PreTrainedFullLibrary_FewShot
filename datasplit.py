#!/usr/bin/env python
# coding: utf-8

import os
from shutil import copy, copytree
import random
import errno
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Split into train, val')
parser.add_argument('data_path', help='path to dataset')

args = parser.parse_args()

data_path = args.data_path
image_path = os.path.join(data_path, 'all')
class_names = [folder for folder in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, folder))]

print('Total number of classes:', len(class_names))

classes = {}

if os.path.exists(os.path.join(data_path, 'classes_train.txt'))\
        and os.path.exists(os.path.join(data_path, 'classes_val.txt')):

    with open(os.path.join(data_path, 'classes_train.txt'), 'r') as f:
        classes['train'] = f.read().splitlines()

    with open(os.path.join(data_path, 'classes_val.txt'), 'r') as f:
        classes['val'] = f.read().splitlines()


else:
    random.seed(1)
    random.shuffle(class_names)
    percentage_train_class = 90
    percentage_val_class = 10
    train_val_ratio = [
        percentage_train_class, percentage_val_class]

    num_val, num_test = [
        int(float(ratio) / np.sum(train_val_ratio) * len(class_names))
        for ratio in train_val_ratio]

    classes = {
        'train': class_names[:num_val],
        'val': class_names[num_val:]
    }

    if not os.path.exists(os.path.join(data_path, 'classes_train.txt')):
        with open(os.path.join(data_path, 'classes_train.txt'), 'w') as f:
            f.writelines("%s\n" % class_name for class_name in classes['train'])

    if not os.path.exists(os.path.join(data_path, 'classes_val.txt')):
        with open(os.path.join(data_path, 'classes_val.txt'), 'w') as f:
            f.writelines("%s\n" % class_name for class_name in classes['val'])


if not os.path.exists(os.path.join(data_path, 'train')):
    os.makedirs(os.path.join(data_path, 'train'))
    for class_name in classes['train']:
        source = os.path.join(image_path, class_name)
        try:
            copytree(source, os.path.join(data_path, 'train', class_name))
        except OSError as e:
            # If the error was caused because the source wasn't a directory, simply ignore
            if e.errno == errno.ENOTDIR:
                pass
            else:
                print('Could not copy directory!')

if not os.path.exists(os.path.join(data_path, 'val')):
    os.makedirs(os.path.join(data_path, 'val'))
    for class_name in classes['val']:
        source = os.path.join(image_path, class_name)
        try:
            copytree(source, os.path.join(data_path, 'val', class_name))
        except OSError as e:
            # If the error was caused because the source wasn't a directory, simply ignore
            if e.errno==errno.ENOTDIR:
                pass
            else:
                print('Could not copy directory!')
import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import glob
from pathlib import Path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath):

    left_fold    = 'left.png'
    right_fold   = 'right.png'
    disp_L       = 'left_disp.png'
    disp_R       = 'right_disp.png'

    glob_path = Path(filepath)
    #print(glob_path)
    left  = list(glob_path.glob('**/' + left_fold))
    right = [p.parent.joinpath(right_fold) for p in left]
    dispL = [p.parent.joinpath(disp_L) for p in left]
    dispR = [p.parent.joinpath(disp_R) for p in left]
    print('Number of images: %d' % len(left))
    #print(right)
    # Split 70:30
    split_count = int(len(left) * 0.7)

    left_train  = left[:split_count]
    right_train = right[:split_count]
    disp_train_L = dispL[:split_count]
    disp_train_R = dispR[:split_count]
    #disp_train_R = [filepath+disp_R+img for img in train]

    left_val  = left[split_count:]
    right_val = right[split_count:]
    disp_val_L = dispL[split_count:]
    disp_val_R = dispR[split_count:]
    #disp_val_R = [filepath+disp_R+img for img in val]

    return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L

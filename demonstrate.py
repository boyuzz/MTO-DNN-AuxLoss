import os
from os.path import join
from os import listdir
import numpy as np
from skimage import io, draw
from skimage.transform import rescale
import cv2
import util
from matplotlib import pyplot as plt


def bbox_expand(im, start, end):
    rr, cc = draw.rectangle(start=start, end=end)
    cv2.rectangle(im, start, end, (255, 0, 0), 2)
    im_bbox = im[cc, rr]
    crop_resized = rescale(im_bbox, im.shape[1]/im_bbox.shape[1], multichannel=True, anti_aliasing=True)
    # io.imshow(crop_resized)
    # io.show()
    im_comb = np.concatenate((im/255., crop_resized), axis=0)
    return im_comb


if __name__ == '__main__':
    # Specify the folder of tested images
    folder = './result/ipiu/1/'
    # Specify the top-left and bottom-right corners of the interested area
    start = (141, 231)
    end = (275, 330)
    # start = (146, 124)
    # end = (272, 234)
    # start = (150, 0)
    # end = (200, 50)
    # start = (200, 240)
    # end = (240, 275)
    hr_list = [join(folder, x) for x in sorted(listdir(folder)) if util.is_image_file(x)]
    for fname in hr_list:
        im = io.imread(fname)
        # plt.imshow(im)
        # plt.show()
        im_expand = bbox_expand(im, start, end)
        name = os.path.basename(fname)
        io.imsave(join(folder, 'bbox_{}'.format(name)), im_expand)

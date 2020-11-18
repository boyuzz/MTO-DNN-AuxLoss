import torch.utils.data as data
import h5py
import torch
import random

import os
from os import listdir
from os.path import join
from skimage import io, transform
import numpy as np
import torchvision.transforms as tfs
from PIL import Image
from imresize import imresize
import util
from skimage import color, io
# from torchsample.transforms import *


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp", ".JPG"])


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        data, target = sample['data'], sample['target']

        h, w = data.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        data = transform.resize(data, (new_h, new_w))

        # h and w are swapped for target because for images,
        # x and y axes are axis 1 and 0 respectively
        target = target * [new_w / w, new_h / h]

        return {'data': data, 'target': target}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        data, target = sample['data'], sample['target']

        h, w = data.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        data = data[top: top + new_h,
                      left: left + new_w]

        target = target[top: top + new_h,
                      left: left + new_w]

        return {'data': data, 'target': target}


class RandomRotate(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __call__(self, sample):
        data, target = sample['data'], sample['target']

        direct = np.random.randint(0, 4)
        data = transform.rotate(data, 90*direct)
        target = transform.rotate(target, 90*direct)

        return {'data': data, 'target': target}


class RandomFlip(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        data, target = sample['data'], sample['target']

        if np.random.rand() > self.p:
            # data = data[:, ::-1]
            # target = target[:, ::-1]
            data = np.fliplr(data)
            target = np.fliplr(target)
            # io.imshow(target)
            # io.show()

        return {'data': data, 'target': target}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, target = sample['data'], sample['target']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        data = data.transpose((2, 0, 1))
        target = target.transpose((2, 0, 1))
        return {'data': torch.from_numpy(data).float(),
                'target': torch.from_numpy(target).float()}


class Transpose(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, target = sample['data'], sample['target']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        data = data.transpose((1, 2, 0))
        target = target.transpose((1, 2, 0))
        return {'data': data,
                'target': target}


def calculate_valid_crop_size(crop_size, scale_factor):
    return crop_size - (crop_size % scale_factor)


class TrainDatasetFromFolder(data.Dataset):
    def __init__(self, image_dirs, is_gray=False, random_scale=True, crop_size=128, rotate=True, fliplr=True,
                 fliptb=True, scale_factor=4, bic_inp = False, preload=True):
        super(TrainDatasetFromFolder, self).__init__()

        self.image_filenames = []
        self.preload = preload
        all_files = os.walk(image_dirs)
        for path, dir_list, file_list in all_files:
            self.image_filenames.extend(join(path, x) for x in file_list if is_image_file(x))
        if self.preload:
            self.image_list = []
            for file in self.image_filenames:
                img = Image.open(file).convert('RGB')
                self.image_list.append(img)
            # self.image_filenames.extend(join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x))
        self.is_gray = is_gray
        self.random_scale = random_scale
        self.crop_size = crop_size
        self.rotate = rotate
        self.fliplr = fliplr
        self.fliptb = fliptb
        self.scale_factor = scale_factor
        self.bic_inp = bic_inp

    def __getitem__(self, index):
        # load image
        if self.preload:
            img = self.image_list[index]
        else:
            img = Image.open(self.image_filenames[index]).convert('RGB')

        # determine valid HR image size with scale factor
        self.crop_size = calculate_valid_crop_size(self.crop_size, self.scale_factor)
        hr_img_w = self.crop_size
        hr_img_h = self.crop_size

        # determine LR image size
        lr_img_w = hr_img_w // self.scale_factor
        lr_img_h = hr_img_h // self.scale_factor

        # random scaling between [0.5, 1.0]
        if self.random_scale:
            eps = 1e-3
            ratio = random.randint(5, 10) * 0.1
            if hr_img_w * ratio < self.crop_size:
                ratio = self.crop_size / hr_img_w + eps
            if hr_img_h * ratio < self.crop_size:
                ratio = self.crop_size / hr_img_h + eps

            scale_w = int(hr_img_w * ratio)
            scale_h = int(hr_img_h * ratio)

            img = np.asarray(img)
            img = imresize(img, output_shape=(scale_h, scale_w))
            img = Image.fromarray(img.squeeze())

        # random crop
        transform = tfs.RandomCrop(self.crop_size)
        img = transform(img)

        # random rotation between [90, 180, 270] degrees
        if self.rotate:
            rv = random.randint(1, 3)
            img = img.rotate(90 * rv, expand=True)

        # random horizontal flip
        if self.fliplr:
            transform = tfs.RandomHorizontalFlip()
            img = transform(img)

        # random vertical flip
        if self.fliptb:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # only Y-channel is super-resolved
        if self.is_gray:
            img = np.asarray(img)
            img = color.rgb2ycbcr(img) / 255
            channel = len(img.shape)
            img, _, _ = np.split(img, indices_or_sections=channel, axis=-1)
            # img = img.convert('YCbCr')
            # precision degrade from float64 to float32
            img = Image.fromarray(img.squeeze())

        # hr_img HR image
        tensor_transform = tfs.ToTensor()
        hr_img = tensor_transform(img)

        # lr_img LR image
        img = np.asarray(img)
        if not self.bic_inp:
            lr_img = imresize(img, output_shape=(lr_img_w, lr_img_h))
        else:
            lr_img = imresize(imresize(img, output_shape=(lr_img_w, lr_img_h)), output_shape=(hr_img_w, hr_img_h))

        lr_img = Image.fromarray(lr_img.squeeze())
        lr_img = tensor_transform(lr_img)

        return lr_img, hr_img#, bc_img

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, is_gray=False, scale_factor=4, bic_inp = 0):
        super(TestDatasetFromFolder, self).__init__()

        self.image_filenames = []
        all_files = os.walk(image_dir)
        for path, dir_list, file_list in all_files:
            self.image_filenames.extend(join(path, x) for x in file_list if is_image_file(x))
        # self.image_filenames = [join(image_dir, x) for x in sorted(listdir(image_dir)) if is_image_file(x)]
        self.is_gray = is_gray
        self.scale_factor = scale_factor
        self.bic_inp = bic_inp

    def __getitem__(self, index):
        # load image
        img = Image.open(self.image_filenames[index]).convert('RGB')
        img = np.asarray(img)
        img = util.modcrop(img, self.scale_factor)
        # original HR image size
        hr_img_w = img.shape[0]
        hr_img_h = img.shape[1]

        # determine lr_img LR image size
        lr_img_w = hr_img_w // self.scale_factor
        lr_img_h = hr_img_h // self.scale_factor

        # only Y-channel is super-resolved
        if self.is_gray:
            img = np.asarray(img)
            img = color.rgb2ycbcr(img) / 255
            channel = len(img.shape)
            img, _, _ = np.split(img, indices_or_sections=channel, axis=-1)
            # img = img.convert('YCbCr')
            # precision degrade from float64 to float32
            img = Image.fromarray(img.squeeze())

        # hr_img HR image
        tensor_transform = tfs.ToTensor()
        hr_img = tensor_transform(img)

        # lr_img LR image
        img = np.asarray(img)
        if not self.bic_inp:
            lr_img = imresize(img, output_shape=(lr_img_w, lr_img_h))
        else:
            lr_img = imresize(imresize(img, output_shape=(lr_img_w, lr_img_h)), output_shape=(hr_img_w, hr_img_h))

        bc_img = imresize(lr_img, output_shape=(hr_img_w, hr_img_h))

        bc_img = Image.fromarray(bc_img.squeeze())
        lr_img = Image.fromarray(lr_img.squeeze())
        bc_img = tensor_transform(bc_img)
        lr_img = tensor_transform(lr_img)

        return lr_img, hr_img, bc_img

    def __len__(self):
        return len(self.image_filenames)


class DatasetFromHdf5Lap(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5Lap, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get("data")
        self.label_x2 = hf.get("label_x2")
        self.label_x4 = hf.get("label_x4")

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.label_x2[index,:,:,:]).float(), torch.from_numpy(self.label_x4[index,:,:,:]).float()

    def __len__(self):
        return self.data.shape[0]


class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path, random_crop=False, random_rotate=False, random_flip=False):  # , random_crop=True, rotate=True, random_flip=True, scale_factor=2
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get('data')
        self.target = hf.get('label')
        self.random_crop = random_crop
        self.random_rotate = random_rotate
        self.random_flip = random_flip
        self.build_transform()

    def build_transform(self):
        operation = [Transpose()]
        if self.random_crop:
            operation.append(RandomCrop(96))

        if self.random_flip:
            operation.append(RandomFlip())

        if self.random_rotate:
            operation.append(RandomRotate())
        operation.append(ToTensor())
        # if self.rotate:
        #     operation.append(transforms.RandomRotation(90))
        self.trans = tfs.Compose(operation)

    def __getitem__(self, index):
        sample = {'data': self.data[index, :, :, :], 'target': self.target[index, :, :, :]}
        # sample = (self.data[index, :, :, :], self.target[index, :, :, :])
        sample = self.trans(sample)
        imgIn, imgTar = sample['data'], sample['target']
        # imgIn, imgTar = sample
        # imgIn = torch.from_numpy(self.data[index, :, :, :]).float()
        # imgTar = torch.from_numpy(self.target[index, :, :, :]).float()

        # # random crop


        # if self.rotate:
        #     rv = random.randint(0, 3)
        #     transform = Rotate(90 * rv)
        #     imgIn = transform(imgIn)
        #     imgTar = transform(imgTar)

        # if self.random_flip:
        #     horizon = random.randint(0, 1)
        #     vertical = random.randint(0, 1)
        #     transform = RandomFlip(horizon, vertical, 1)
        #     imgIn = transform(imgIn)
        #     imgTar = transform(imgTar)

        return imgIn, imgTar

    def __len__(self):
        return self.data.shape[0]


def getPatch(imgIn, imgTar, patchSize, scale):
    (_, ih, iw, c) = imgIn.shape
    # (th, tw) = (scale * ih, scale * iw)

    tp = scale * patchSize
    ip = patchSize

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    (tx, ty) = (scale * ix, scale * iy)
    imgIn = imgIn[:, iy:iy + ip, ix:ix + ip, :]
    imgTar = imgTar[:, ty:ty + tp, tx:tx + tp, :]
    patchInfo = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return imgIn, imgTar, patchInfo